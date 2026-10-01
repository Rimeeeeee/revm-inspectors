use crate::{
    opcode::immediate_size,
    tracing::{
        arena::PushTraceKind,
        types::{
            CallKind, CallTraceNode, RecordedMemory, StorageChange, StorageChangeReason,
            TraceMemberOrder,
        },
        utils::gas_used,
    },
};
use alloc::{boxed::Box, vec::Vec};
use alloy_eip8141::{Frame, FrameMode, FrameStatus, ENTRY_POINT};
use core::{borrow::Borrow, mem};
use revm::{
    bytecode::opcode::{self, OpCode},
    context::{JournalTr, LocalContextTr},
    context_interface::{result::ExecutionResult, transaction::Transaction, Cfg, ContextTr},
    handler::FrameResult,
    inspector::JournalExt,
    interpreter::{
        interpreter_action::FrameInput,
        interpreter_types::{Immediates, Jumps, LoopControl, ReturnData, RuntimeFlag},
        CallInput, CallInputs, CallOutcome, CallScheme, CreateInputs, CreateOutcome, Interpreter,
        InterpreterResult,
    },
    primitives::{hardfork::SpecId, Address, Bytes, Log, B256, U256},
    Inspector, JournalEntry,
};

mod arena;
pub use arena::CallTraceArena;

mod builder;
pub use builder::{
    geth::{self, GethTraceBuilder},
    parity::{self, ParityTraceBuilder},
};

mod config;
pub use config::{OpcodeFilter, StackSnapshotType, TracingInspectorConfig};

mod fourbyte;
pub use fourbyte::FourByteInspector;

mod opcount;
pub use opcount::OpcodeCountInspector;

pub mod types;
use types::{CallLog, CallTrace, CallTraceStep};

mod utils;

#[cfg(feature = "std")]
mod writer;
#[cfg(feature = "std")]
pub use writer::{TraceWriter, TraceWriterConfig};

#[cfg(feature = "js-tracer")]
pub mod js;

mod mux;
pub use mux::{Error as MuxError, MuxInspector};

mod debug;
pub use debug::{DebugInspector, DebugInspectorError};

/// An inspector that collects call traces.
///
/// This [Inspector] can be hooked into revm's EVM which then calls the inspector
/// functions, such as [Inspector::call] or [Inspector::call_end].
///
/// The [TracingInspector] keeps track of everything by:
///   1. start tracking steps/calls on [Inspector::step] and [Inspector::call]
///   2. complete steps/calls on [Inspector::step_end] and [Inspector::call_end]
#[derive(Clone, Debug, Default)]
pub struct TracingInspector {
    /// Configures what and how the inspector records traces.
    config: TracingInspectorConfig,
    /// Records all call traces
    traces: CallTraceArena,
    /// Tracks active calls
    trace_stack: Vec<usize>,
    /// Tracks whether the next `step_end` should be recorded. Set in `start_step`.
    record_step_end: bool,
    /// Number of logs recorded so far, used as the index of the next log.
    log_count: usize,
    /// Number of opcode steps captured across all calls since the last reset.
    recorded_steps: u64,
    /// Tracks the journal len in the step, used in step_end to check if the journal has changed
    last_journal_len: usize,
    /// The spec id of the EVM.
    ///
    /// This is filled during execution.
    spec_id: Option<SpecId>,
    /// Pool of reusable _empty_ step vectors to reduce allocations.
    ///
    /// All `Vec<CallTraceStep>` are always empty but may have capacity.
    reusable_step_vecs: Vec<Vec<CallTraceStep>>,
    /// EIP-8141 frame-transaction bookkeeping retained until receipt finalization.
    frame_transaction: Option<FrameTransactionTrace>,
}

/// EIP-8141 data which is unavailable to ordinary EVM call hooks.
///
/// In particular, skipped atomic-batch frames and final state-gas/log accounting are only known
/// after the frame transaction has completed.
#[derive(Clone, Debug)]
struct FrameTransactionTrace {
    /// Synthetic root trace node.
    root: usize,
    /// Transaction sender, used to resolve empty frame targets.
    sender: Address,
    /// Original ordered frame definitions.
    frames: Vec<Frame>,
    /// Arena node for every frame which entered the frame executor.
    frame_nodes: Vec<Option<usize>>,
    /// Top-level frame whose call hook is about to run.
    pending_frame: Option<usize>,
}

/// An error returned while finalizing an EIP-8141 frame-transaction trace.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum FrameTransactionTraceError {
    /// The supplied execution result is not for a frame transaction.
    #[error("execution result is not an EIP-8141 frame transaction")]
    NotFrameTransaction,
    /// The inspector did not observe any executable frame in the transaction.
    #[error("tracing inspector did not observe the EIP-8141 frame transaction")]
    NotObserved,
    /// The execution result does not contain one receipt and output per frame.
    #[error(
        "EIP-8141 result length mismatch: {frames} frames, {receipts} receipts, {outputs} outputs"
    )]
    ResultLengthMismatch {
        /// Number of frames in the transaction.
        frames: usize,
        /// Number of frame receipts in the execution result.
        receipts: usize,
        /// Number of frame outputs in the execution result.
        outputs: usize,
    },
}

impl TracingInspector {
    /// Returns a new instance for the given config
    pub fn new(config: TracingInspectorConfig) -> Self {
        Self { config, ..Default::default() }
    }

    /// Resets the inspector to its initial state of [Self::new].
    /// This makes the inspector ready to be used again.
    ///
    /// Note that this method has no effect on the allocated capacity of the vector.
    #[inline]
    pub fn fuse(&mut self) {
        let Self {
            traces,
            trace_stack,
            log_count,
            last_journal_len,
            spec_id,
            record_step_end,
            recorded_steps,
            // kept
            config,
            reusable_step_vecs,
            frame_transaction,
        } = self;

        // if we record steps we can reuse the individual calltracestep vecs
        if config.record_steps {
            for node in &mut traces.arena {
                // move out and store the reusable steps vec
                let mut steps = mem::take(&mut node.trace.steps);
                // ensure steps are cleared
                steps.clear();
                reusable_step_vecs.push(steps);
            }
        }

        traces.clear();
        trace_stack.clear();
        spec_id.take();
        *log_count = 0;
        *last_journal_len = 0;
        *record_step_end = false;
        *recorded_steps = 0;
        frame_transaction.take();
    }

    /// Resets the inspector to it's initial state of [Self::new].
    #[inline]
    pub fn fused(mut self) -> Self {
        self.fuse();
        self
    }

    /// Returns the config of the inspector.
    pub const fn config(&self) -> &TracingInspectorConfig {
        &self.config
    }

    /// Returns a mutable reference to the config of the inspector.
    pub fn config_mut(&mut self) -> &mut TracingInspectorConfig {
        &mut self.config
    }

    /// Updates the config of the inspector.
    pub fn update_config(
        &mut self,
        f: impl FnOnce(TracingInspectorConfig) -> TracingInspectorConfig,
    ) {
        self.config = f(self.config);
    }

    /// Gets a reference to the recorded call traces.
    pub const fn traces(&self) -> &CallTraceArena {
        &self.traces
    }

    #[doc(hidden)]
    #[deprecated = "use `traces` instead"]
    pub const fn get_traces(&self) -> &CallTraceArena {
        &self.traces
    }

    /// Gets a mutable reference to the recorded call traces.
    pub fn traces_mut(&mut self) -> &mut CallTraceArena {
        &mut self.traces
    }

    #[doc(hidden)]
    #[deprecated = "use `traces_mut` instead"]
    pub fn get_traces_mut(&mut self) -> &mut CallTraceArena {
        &mut self.traces
    }

    /// Consumes the inspector and returns the recorded call traces.
    pub fn into_traces(self) -> CallTraceArena {
        self.traces
    }

    /// Manually set the gas used of the root trace.
    ///
    /// This is useful if the root trace's gasUsed should mirror the actual gas used by the
    /// transaction.
    ///
    /// This allows setting it manually by consuming the execution result's gas for example.
    #[inline]
    pub fn set_transaction_gas_used(&mut self, gas_used: u64) {
        if let Some(node) = self.traces.arena.first_mut() {
            node.trace.gas_used = gas_used;
        }
    }

    /// Manually set the gas limit of the debug root trace.
    ///
    /// This is useful if the debug root trace's gasUsed should mirror the actual gas used by the
    /// transaction.
    ///
    /// This allows setting it manually by consuming the execution result's gas for example.
    #[inline]
    pub fn set_transaction_gas_limit(&mut self, gas_limit: u64) {
        if let Some(node) = self.traces.arena.first_mut() {
            node.trace.gas_limit = gas_limit;
        }
    }

    /// Convenience function for [ParityTraceBuilder::set_transaction_gas_used] that consumes the
    /// type.
    #[inline]
    pub fn with_transaction_gas_used(mut self, gas_used: u64) -> Self {
        self.set_transaction_gas_used(gas_used);
        self
    }

    /// Work with [TracingInspector::set_transaction_gas_limit] function
    #[inline]
    pub fn with_transaction_gas_limit(mut self, gas_limit: u64) -> Self {
        self.set_transaction_gas_limit(gas_limit);
        self
    }

    /// Manually set the caller address of the root trace.
    ///
    /// This is useful for custom transaction types (e.g. account abstraction batches) where the
    /// EVM's call entry point may not reflect the actual transaction sender.
    #[inline]
    pub fn set_transaction_caller(&mut self, caller: Address) {
        if let Some(node) = self.traces.arena.first_mut() {
            node.trace.caller = caller;
        }
    }

    /// Finalizes an EIP-8141 frame trace from the canonical execution result.
    ///
    /// EIP-8141 executes each top-level frame separately. The execution result is therefore the
    /// source of truth for each frame's final gas, output, and receipt status, including frames
    /// skipped after an atomic batch failure. Call this once, after `transact` returns its
    /// [`ExecutionResult::FrameTransaction`].
    ///
    /// Frames that fail before an inspector hook runs are reconstructed from their transaction
    /// data and receipt so that every transaction frame is represented in the trace.
    pub fn finalize_frame_transaction<HaltReasonTy>(
        &mut self,
        result: &ExecutionResult<HaltReasonTy>,
    ) -> Result<(), FrameTransactionTraceError> {
        let ExecutionResult::FrameTransaction { frame_receipts, frame_outputs, .. } = result else {
            return Err(FrameTransactionTraceError::NotFrameTransaction);
        };
        let Some(frame_transaction) = self.frame_transaction.as_ref() else {
            return Err(FrameTransactionTraceError::NotObserved);
        };

        let root = frame_transaction.root;
        let sender = frame_transaction.sender;
        let frames = frame_transaction.frames.clone();
        let mut frame_nodes = frame_transaction.frame_nodes.clone();

        if frame_receipts.len() != frames.len() || frame_outputs.len() != frames.len() {
            return Err(FrameTransactionTraceError::ResultLengthMismatch {
                frames: frames.len(),
                receipts: frame_receipts.len(),
                outputs: frame_outputs.len(),
            });
        }

        for (index, (frame, receipt)) in frames.iter().zip(frame_receipts).enumerate() {
            let node = match frame_nodes[index] {
                Some(node) => node,
                None => {
                    let node = self.push_unexecuted_frame_trace(
                        root,
                        sender,
                        frame,
                        index,
                        receipt.status,
                    );
                    frame_nodes[index] = Some(node);
                    node
                }
            };

            let trace = &mut self.traces.arena[node].trace;
            trace.gas_limit = frame.limits.execution.saturating_add(frame.limits.state);
            trace.gas_used = receipt.gas_used.execution.saturating_add(receipt.gas_used.state);
            trace.output = frame_outputs[index].clone();
            trace.frame_index = Some(index);

            match receipt.status {
                FrameStatus::Success => {
                    trace.success = true;
                    trace.error = None;
                    trace.status = Some(revm::interpreter::InstructionResult::Stop);
                    if receipt.logs.is_empty() {
                        self.clear_trace_logs(node);
                    }
                }
                FrameStatus::Failure => {
                    trace.success = false;
                    if trace.status.is_none_or(|status| status.is_ok()) {
                        trace.error = Some("frame failed".into());
                    }
                    self.clear_trace_logs(node);
                }
                FrameStatus::SkippedAtomicBatch => {
                    trace.success = false;
                    trace.status = None;
                    trace.error = Some("frame skipped".into());
                    trace.output = Bytes::new();
                    trace.gas_used = 0;
                    self.clear_trace_logs(node);
                }
            }
        }

        let children = {
            let nodes = &self.traces.arena;
            let mut children = frame_nodes
                .iter()
                .copied()
                .collect::<Option<Vec<_>>>()
                .expect("all EIP-8141 frames have a trace node after finalization");
            children.sort_unstable_by_key(|node| {
                nodes[*node].trace.frame_index.expect("frame trace nodes carry their frame index")
            });
            children
        };
        let root_node = &mut self.traces.arena[root];
        root_node.children = children;
        root_node.ordering = (0..root_node.children.len()).map(TraceMemberOrder::Call).collect();
        root_node.trace.gas_used = result.tx_gas_used();
        root_node.trace.output = Bytes::new();
        root_node.trace.success = true;
        root_node.trace.status = Some(revm::interpreter::InstructionResult::Stop);
        root_node.trace.error = None;

        let mut log_index = 0;
        for child in root_node.children.clone() {
            self.reindex_trace_logs(child, &mut log_index);
        }

        if self.trace_stack.last().copied() == Some(root) {
            self.trace_stack.pop();
        }
        if let Some(frame_transaction) = self.frame_transaction.as_mut() {
            frame_transaction.frame_nodes = frame_nodes;
            frame_transaction.pending_frame = None;
        }
        Ok(())
    }

    /// Consumes the Inspector and returns a [ParityTraceBuilder].
    #[inline]
    pub fn into_parity_builder(self) -> ParityTraceBuilder {
        ParityTraceBuilder::new(self.traces.arena, self.spec_id, self.config)
    }

    /// Consumes the Inspector and returns a [GethTraceBuilder].
    #[inline]
    pub fn into_geth_builder(self) -> GethTraceBuilder<'static> {
        let builder = GethTraceBuilder::new(self.traces.arena);
        match self.spec_id {
            Some(spec_id) => builder.with_spec_id(spec_id),
            None => builder,
        }
    }

    /// Returns the  [GethTraceBuilder] for the recorded traces without consuming the type.
    ///
    /// This can be useful for multiple transaction tracing (block) where this inspector can be
    /// reused for each transaction but caller must ensure that the traces are cleared before
    /// starting a new transaction: [`Self::fuse`]
    #[inline]
    pub fn geth_builder(&self) -> GethTraceBuilder<'_> {
        let builder = GethTraceBuilder::new_borrowed(&self.traces.arena);
        match self.spec_id {
            Some(spec_id) => builder.with_spec_id(spec_id),
            None => builder,
        }
    }

    /// Returns true if we're no longer in the context of the root call.
    fn is_deep(&self) -> bool {
        // the root call will always be the first entry in the trace stack
        self.trace_stack.len() > usize::from(self.frame_transaction.is_some())
    }

    /// Returns true if this a call to a precompile contract.
    ///
    /// Returns true if the `to` address is a precompile contract and the value is zero.
    #[inline]
    fn is_precompile_call<CTX: ContextTr<Journal: JournalExt>>(
        &self,
        context: &CTX,
        to: &Address,
        value: &U256,
    ) -> bool {
        if context.journal_ref().precompile_addresses().contains(to) {
            // only if this is _not_ the root call
            return self.is_deep() && value.is_zero();
        }
        false
    }

    /// Returns the currently active call trace.
    ///
    /// This will be the last call trace pushed to the stack: the call we entered most recently.
    #[track_caller]
    #[inline]
    fn active_trace(&self) -> Option<&CallTraceNode> {
        self.trace_stack.last().map(|idx| &self.traces.arena[*idx])
    }

    /// Returns the last trace [CallTrace] index from the stack.
    ///
    /// This will be the currently active call trace.
    ///
    /// # Panics
    ///
    /// If no [CallTrace] was pushed
    #[track_caller]
    #[inline]
    fn last_trace_idx(&self) -> usize {
        self.trace_stack.last().copied().expect("can't start step without starting a trace first")
    }

    /// Returns a mutable reference to the last trace [CallTrace] from the stack.
    #[track_caller]
    fn last_trace(&mut self) -> &mut CallTraceNode {
        let idx = self.last_trace_idx();
        &mut self.traces.arena[idx]
    }

    /// _Removes_ the last trace [CallTrace] index from the stack.
    ///
    /// # Panics
    ///
    /// If no [CallTrace] was pushed
    #[track_caller]
    #[inline]
    fn pop_trace_idx(&mut self) -> usize {
        self.trace_stack.pop().expect("more traces were filled than started")
    }

    /// Starts tracking a new trace.
    ///
    /// Invoked on [Inspector::call].
    #[allow(clippy::too_many_arguments)]
    fn start_trace_on_call<CTX: ContextTr>(
        &mut self,
        context: &mut CTX,
        address: Address,
        input_data: Bytes,
        value: U256,
        kind: CallKind,
        caller: Address,
        gas_limit: u64,
        maybe_precompile: Option<bool>,
    ) -> usize {
        // This will only be true if the inspector is configured to exclude precompiles and the call
        // is to a precompile
        let push_kind = if maybe_precompile.unwrap_or(false) {
            // We don't want to track precompiles
            PushTraceKind::PushOnly
        } else {
            PushTraceKind::PushAndAttachToParent
        };

        // find an empty steps vec or create a new one
        let steps = self.reusable_step_vecs.pop().unwrap_or_default();

        // the currently active call is the parent of the new call
        let parent = self.trace_stack.last().copied().unwrap_or_default();

        let trace = self.traces.push_trace(
            parent,
            push_kind,
            CallTrace {
                depth: context.journal().depth(),
                address,
                kind,
                data: input_data,
                value,
                status: None,
                caller,
                maybe_precompile,
                gas_limit,
                steps,
                ..Default::default()
            },
        );
        self.trace_stack.push(trace);
        trace
    }

    /// Starts the synthetic EIP-8141 transaction root before the first top-level frame call.
    fn start_frame_transaction_trace<CTX: ContextTr>(&mut self, context: &CTX) {
        let transaction = context
            .tx()
            .frame_transaction()
            .expect("frame transaction runtime implies a frame transaction");
        let root = self.traces.push_trace(
            0,
            PushTraceKind::PushAndAttachToParent,
            CallTrace {
                depth: 0,
                success: true,
                caller: context.tx().caller(),
                address: ENTRY_POINT,
                kind: CallKind::Call,
                gas_limit: context.tx().gas_limit(),
                frame_transaction_root: true,
                ..Default::default()
            },
        );
        self.trace_stack.push(root);
        self.frame_transaction = Some(FrameTransactionTrace {
            root,
            sender: context.tx().caller(),
            frames: transaction.frames.clone(),
            frame_nodes: vec![None; transaction.frames.len()],
            pending_frame: None,
        });
    }

    /// Returns whether a frame hook is beginning a top-level EIP-8141 frame.
    fn is_top_level_frame<CTX: ContextTr>(&self, context: &CTX) -> bool {
        // The frame orchestrator opens a rollback checkpoint before invoking the inspector. A
        // top-level frame therefore starts at depth one; nested EVM calls start at greater depths.
        context.local().frame_transaction().is_some() && context.journal().depth() == 1
    }

    /// Adds an EIP-8141 frame which did not reach an inspector call hook.
    fn push_unexecuted_frame_trace(
        &mut self,
        root: usize,
        sender: Address,
        frame: &Frame,
        frame_index: usize,
        status: FrameStatus,
    ) -> usize {
        let error = match status {
            FrameStatus::Failure => "frame failed",
            FrameStatus::SkippedAtomicBatch => "frame skipped",
            FrameStatus::Success => "frame trace unavailable",
        };
        self.traces.push_trace(
            root,
            PushTraceKind::PushAndAttachToParent,
            CallTrace {
                depth: 1,
                success: false,
                caller: if frame.mode == FrameMode::Sender { sender } else { ENTRY_POINT },
                address: frame.resolved_target(sender),
                kind: if frame.mode == FrameMode::Verify {
                    CallKind::StaticCall
                } else {
                    CallKind::Call
                },
                value: frame.value,
                data: frame.data.clone(),
                gas_limit: frame.limits.execution.saturating_add(frame.limits.state),
                error: Some(error.into()),
                frame_index: Some(frame_index),
                ..Default::default()
            },
        )
    }

    /// Removes logs from a trace and all of its descendants after a frame-level rollback.
    fn clear_trace_logs(&mut self, trace: usize) {
        let mut pending = vec![trace];
        while let Some(trace) = pending.pop() {
            let node = &mut self.traces.arena[trace];
            pending.extend_from_slice(&node.children);
            node.logs.clear();
            node.ordering.retain(|member| !matches!(member, TraceMemberOrder::Log(_)));
        }
    }

    /// Renumbers visible logs after frame-level rollbacks have removed earlier logs.
    fn reindex_trace_logs(&mut self, trace: usize, log_index: &mut u64) {
        let mut pending = vec![(trace, 0)];
        while let Some((trace, position)) = pending.pop() {
            let Some(member) = self.traces.arena[trace].ordering.get(position).copied() else {
                continue;
            };
            pending.push((trace, position + 1));
            match member {
                TraceMemberOrder::Log(index) => {
                    self.traces.arena[trace].logs[index].index = *log_index;
                    *log_index += 1;
                }
                TraceMemberOrder::Call(index) => {
                    let child = self.traces.arena[trace].children[index];
                    pending.push((child, 0));
                }
                TraceMemberOrder::Step(_) => {}
            }
        }
    }

    /// Fills the current trace with the outcome of a call.
    ///
    /// Invoked on [Inspector::call_end].
    ///
    /// # Panics
    ///
    /// This expects an existing trace [Self::start_trace_on_call]
    fn fill_trace_on_call_end(
        &mut self,
        result: &InterpreterResult,
        created_address: Option<Address>,
    ) {
        let InterpreterResult { result, ref output, ref gas } = *result;

        let trace_idx = self.pop_trace_idx();
        let trace = &mut self.traces.arena[trace_idx].trace;

        trace.gas_used = gas.total_gas_spent();
        trace.gas_refund_counter = gas.refunded().max(0) as u64;

        trace.status = Some(result);
        trace.success = trace.status.is_some_and(|status| status.is_ok());
        trace.output = output.clone();

        if let Some(address) = created_address {
            // A new contract was created via CREATE
            trace.address = address;
        }
    }

    /// Starts tracking a step
    ///
    /// Invoked on [Inspector::step]
    ///
    /// # Panics
    ///
    /// This expects an existing [CallTrace], in other words, this panics if not within the context
    /// of a call.
    #[cold]
    fn start_step<CTX: ContextTr<Journal: JournalExt>>(
        &mut self,
        interp: &mut Interpreter,
        context: &mut CTX,
    ) {
        // We always want an OpCode, even it is unknown because it could be an additional opcode
        // that not a known constant.
        let op = OpCode::new_or_unknown(interp.bytecode.opcode());

        let record = self.config.should_record_opcode(op)
            && self.config.step_limit.is_none_or(|limit| self.recorded_steps < limit.get());
        self.record_step_end = record;
        if !record {
            return;
        }

        self.recorded_steps += 1;
        let trace_idx = self.last_trace_idx();
        let node = &mut self.traces.arena[trace_idx];

        // Reuse the memory from the previous step if:
        // - there is not opcode filter -- in this case we cannot rely on the order of steps
        // - it exists and has not modified memory
        let memory = self.config.record_memory_snapshots.then(|| {
            if self.config.record_opcodes_filter.is_none() {
                if let Some(prev) = node.trace.steps.last() {
                    if !prev.op.modifies_memory() {
                        if let Some(memory) = &prev.memory {
                            return memory.clone();
                        }
                    }
                }
            }
            RecordedMemory::new(&interp.memory.borrow().context_memory())
        });

        let stack = if self.config.record_stack_snapshots.is_all()
            || self.config.record_stack_snapshots.is_full()
        {
            Some(interp.stack.data().as_slice().into())
        } else {
            None
        };
        let returndata = if self.config.record_returndata_snapshots {
            interp.return_data.buffer().clone()
        } else {
            Bytes::new()
        };

        let gas_used = gas_used(
            interp.runtime_flag.spec_id(),
            interp.gas.total_gas_spent(),
            interp.gas.refunded() as u64,
        );

        let mut immediate_bytes = None;
        if self.config.record_immediate_bytes {
            let size = immediate_size(&interp.bytecode);
            if size != 0 {
                immediate_bytes = Some(Bytes::copy_from_slice(
                    &interp.bytecode.read_slice(size as usize + 1)[1..],
                ));
            }
        }

        self.last_journal_len = context.journal_ref().journal().len();

        let step_idx = node.trace.steps.len();
        node.trace.steps.push(CallTraceStep {
            pc: interp.bytecode.pc(),
            op,
            stack,
            memory,
            returndata,
            gas_remaining: interp.gas.remaining(),
            gas_refund_counter: interp.gas.refunded() as u64,
            gas_used,
            immediate_bytes,
            state_gas_cost: None,
            state_gas_reservoir: SpecId::is_enabled_in(
                interp.runtime_flag.spec_id(),
                SpecId::AMSTERDAM,
            )
            .then_some(interp.gas.reservoir()),
            state_gas_spent: interp.gas.state_gas_spent(),

            // These fields will be populated in `step_end`.
            push_stack: None,
            gas_cost: 0,
            storage_change: None,
            status: None,

            // This is never populated in `TracingInspector`.
            decoded: None,
        });

        node.ordering.push(TraceMemberOrder::Step(step_idx));
    }

    /// Fills the current trace with the output of a step.
    ///
    /// Invoked on [Inspector::step_end].
    #[cold]
    fn fill_step_on_step_end<CTX: ContextTr<Journal: JournalExt>>(
        &mut self,
        interp: &mut Interpreter,
        context: &mut CTX,
    ) {
        // No need to reset here, since it is only read here and it will be overwritten by the next
        // step.
        if !self.record_step_end {
            return;
        }

        let trace_idx = self.last_trace_idx();
        let node = &mut self.traces.arena[trace_idx];
        let step = node.trace.steps.last_mut().unwrap();

        // See comments in `start_step`.
        debug_assert!(
            step.push_stack.is_none()
                && step.gas_cost == 0
                && step.storage_change.is_none()
                && step.status.is_none()
                && step.decoded.is_none(),
            "step in step_end is already filled: {trace_idx} -> {step:#?}",
        );

        if self.config.record_stack_snapshots.is_all()
            || self.config.record_stack_snapshots.is_pushes()
        {
            let outputs = if step.op.is_valid() { step.op.outputs() as usize } else { 0 };
            step.push_stack = Some(
                interp
                    .stack
                    .data()
                    .get(interp.stack.len().saturating_sub(outputs)..)
                    .unwrap_or_default()
                    .into(),
            );
        }

        let journal = context.journal_ref().journal();

        // If journal has not changed, there is no state change to be recorded.
        if self.config.record_state_diff && journal.len() != self.last_journal_len {
            let op = step.op.get();

            step.storage_change = if matches!(op, opcode::SLOAD | opcode::SSTORE) {
                let reason = match op {
                    opcode::SLOAD => StorageChangeReason::SLOAD,
                    opcode::SSTORE => StorageChangeReason::SSTORE,
                    _ => unreachable!(),
                };

                match journal.last() {
                    Some(JournalEntry::StorageChanged { address, key, had_value }) => {
                        // SAFETY: (Address,key) exists if part if StorageChange
                        let value =
                            context.journal_ref().evm_state()[address].storage[key].present_value();
                        let change =
                            StorageChange { key: *key, value, had_value: Some(*had_value), reason };
                        Some(Box::new(change))
                    }
                    Some(JournalEntry::StorageWarmed { key, address }) => {
                        // SAFETY: (Address,key) exists if part if StorageChange
                        let value =
                            context.journal_ref().evm_state()[address].storage[key].present_value();
                        let change = StorageChange { key: *key, value, had_value: None, reason };
                        Some(Box::new(change))
                    }
                    _ => None,
                }
            } else {
                None
            };
        }

        // The gas cost is the difference between the recorded gas remaining at the start of the
        // step the remaining gas here, at the end of the step.
        // TODO: Figure out why this can overflow. https://github.com/paradigmxyz/revm-inspectors/pull/38
        step.gas_cost = step.gas_remaining.saturating_sub(interp.gas.remaining());
        let state_gas_delta = interp.gas.state_gas_spent().saturating_sub(step.state_gas_spent);
        if step.state_gas_reservoir.is_some() && state_gas_delta != 0 {
            step.state_gas_cost = Some(state_gas_delta);
        }

        // set the status
        step.status = interp.bytecode.action().as_ref().and_then(|i| i.instruction_result())
    }
}

impl<CTX> Inspector<CTX> for TracingInspector
where
    CTX: ContextTr<Journal: JournalExt>,
{
    fn initialize_interp(&mut self, interp: &mut Interpreter, _context: &mut CTX) {
        if self.spec_id.is_none() {
            self.spec_id = Some(interp.runtime_flag.spec_id());
        }
        if let Some(trace) = self.trace_stack.last().copied() {
            self.traces.arena[trace].trace.entered_evm = true;
        }
    }

    #[inline]
    fn step(&mut self, interp: &mut Interpreter, context: &mut CTX) {
        if self.config.record_steps {
            self.start_step(interp, context);
        }
    }

    #[inline]
    fn step_end(&mut self, interp: &mut Interpreter, context: &mut CTX) {
        if self.config.record_steps {
            self.fill_step_on_step_end(interp, context);
        }
    }

    fn log(&mut self, _context: &mut CTX, log: Log) {
        if self.config.record_logs {
            // index starts at 0
            let log_count = self.log_count;
            self.log_count += 1;
            let trace = self.last_trace();
            trace.ordering.push(TraceMemberOrder::Log(trace.logs.len()));
            trace.logs.push(
                CallLog::from(log)
                    .with_position(trace.children.len() as u64)
                    .with_index(log_count as u64),
            );
        }
    }

    fn call(&mut self, context: &mut CTX, inputs: &mut CallInputs) -> Option<CallOutcome> {
        if self.spec_id.is_none() {
            self.spec_id = Some(context.cfg().spec().into());
        }

        // determine correct `from` and `to` based on the call scheme
        let (from, to) = match inputs.scheme {
            CallScheme::DelegateCall | CallScheme::CallCode => {
                (inputs.target_address, inputs.bytecode_address)
            }
            _ => (inputs.caller, inputs.target_address),
        };

        let value = if matches!(inputs.scheme, CallScheme::DelegateCall) {
            // for delegate calls we need to use the value of the top trace
            if let Some(parent) = self.active_trace() {
                parent.trace.value
            } else {
                inputs.call_value()
            }
        } else {
            inputs.call_value()
        };

        // if calls to precompiles should be excluded, check whether this is a call to a precompile
        let maybe_precompile = self
            .config
            .exclude_precompile_calls
            .then(|| self.is_precompile_call(context, &to, &value));

        let input = inputs.input_data(context);
        let trace = self.start_trace_on_call(
            context,
            to,
            input,
            value,
            inputs.scheme.into(),
            from,
            inputs.gas_limit,
            maybe_precompile,
        );
        if let Some(frame_transaction) = self.frame_transaction.as_mut() {
            if let Some(frame) = frame_transaction.pending_frame.take() {
                frame_transaction.frame_nodes[frame] = Some(trace);
                self.traces.arena[trace].trace.frame_index = Some(frame);
            }
        }

        None
    }

    fn call_end(&mut self, _: &mut CTX, _inputs: &CallInputs, outcome: &mut CallOutcome) {
        self.fill_trace_on_call_end(&outcome.result, None);
    }

    fn create(&mut self, context: &mut CTX, inputs: &mut CreateInputs) -> Option<CreateOutcome> {
        if self.spec_id.is_none() {
            self.spec_id = Some(context.cfg().spec().into());
        }

        let nonce = context.journal_mut().load_account(inputs.caller()).ok()?.info.nonce;
        self.start_trace_on_call(
            context,
            inputs.created_address(nonce),
            inputs.init_code().clone(),
            inputs.value(),
            inputs.scheme().into(),
            inputs.caller(),
            inputs.gas_limit(),
            Some(false),
        );
        None
    }

    fn create_end(
        &mut self,
        _context: &mut CTX,
        _inputs: &CreateInputs,
        outcome: &mut CreateOutcome,
    ) {
        self.fill_trace_on_call_end(&outcome.result, outcome.address);
    }

    fn selfdestruct(&mut self, contract: Address, target: Address, value: U256) {
        let node = self.last_trace();
        node.trace.selfdestruct_address = Some(contract);
        node.trace.selfdestruct_refund_target = Some(target);
        node.trace.selfdestruct_transferred_value = Some(value);
    }

    fn frame_start(
        &mut self,
        context: &mut CTX,
        _frame_input: &mut FrameInput,
    ) -> Option<FrameResult> {
        if !self.is_top_level_frame(context) {
            return None;
        }
        if self.frame_transaction.is_none() {
            self.start_frame_transaction_trace(context);
        }
        let frame_index = context
            .local()
            .frame_transaction()
            .expect("top-level frame has frame runtime")
            .current_frame_index;
        self.frame_transaction
            .as_mut()
            .expect("frame transaction trace was initialized")
            .pending_frame = Some(frame_index);
        None
    }
}

/// Contains some contextual infos for a transaction execution that is made available to the JS
/// object.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TransactionContext {
    /// Hash of the block the tx is contained within.
    ///
    /// `None` if this is a call.
    pub block_hash: Option<B256>,
    /// Index of the transaction within a block.
    ///
    /// `None` if this is a call.
    pub tx_index: Option<usize>,
    /// Hash of the transaction being traced.
    ///
    /// `None` if this is a call.
    pub tx_hash: Option<B256>,
}

impl TransactionContext {
    /// Sets the block hash.
    pub const fn with_block_hash(mut self, block_hash: B256) -> Self {
        self.block_hash = Some(block_hash);
        self
    }

    /// Sets the index of the transaction within a block.
    pub const fn with_tx_index(mut self, tx_index: usize) -> Self {
        self.tx_index = Some(tx_index);
        self
    }

    /// Sets the hash of the transaction.
    pub const fn with_tx_hash(mut self, tx_hash: B256) -> Self {
        self.tx_hash = Some(tx_hash);
        self
    }
}

impl From<alloy_rpc_types_eth::TransactionInfo> for TransactionContext {
    fn from(tx_info: alloy_rpc_types_eth::TransactionInfo) -> Self {
        Self {
            block_hash: tx_info.block_hash,
            tx_index: tx_info.index.map(|idx| idx as usize),
            tx_hash: tx_info.hash,
        }
    }
}

/// A helper extension trait that _clones_ the input data from the shared mem buffer
pub(crate) trait CallInputExt {
    fn input_data<CTX: ContextTr>(&self, ctx: &mut CTX) -> Bytes;
}

impl CallInputExt for CallInputs {
    fn input_data<CTX: ContextTr>(&self, ctx: &mut CTX) -> Bytes {
        match &self.input {
            CallInput::SharedBuffer(range) => ctx
                .local()
                .shared_memory_buffer_slice(range.clone())
                .map(|slice| Bytes::copy_from_slice(&slice))
                .unwrap_or_default(),
            CallInput::Bytes(bytes) => bytes.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy_eip8141::{FrameGasUsed, FrameLimits, FrameReceipt};
    use alloy_rpc_types_trace::geth::CallConfig;
    use revm::context_interface::result::ResultGas;

    fn frame(target: Address) -> Frame {
        Frame {
            target: target.into(),
            limits: FrameLimits { execution: 10, state: 10 },
            ..Default::default()
        }
    }

    fn receipt(status: FrameStatus, execution: u64, state: u64) -> FrameReceipt<Log> {
        FrameReceipt { status, gas_used: FrameGasUsed { execution, state }, logs: Vec::new() }
    }

    fn frame_node(address: Address, frame_index: usize) -> CallTrace {
        CallTrace {
            depth: 1,
            success: true,
            status: Some(revm::interpreter::InstructionResult::Stop),
            address,
            kind: CallKind::Call,
            gas_limit: 10,
            frame_index: Some(frame_index),
            entered_evm: true,
            ..Default::default()
        }
    }

    #[test]
    fn finalizes_frame_receipts_and_inserts_skipped_frames_in_order() {
        let sender = Address::with_last_byte(1);
        let addresses = [
            Address::with_last_byte(2),
            Address::with_last_byte(3),
            Address::with_last_byte(4),
            Address::with_last_byte(5),
        ];
        let frames = addresses.into_iter().map(frame).collect::<Vec<_>>();
        let mut inspector = TracingInspector::default();
        inspector.traces.arena[0].trace = CallTrace {
            success: true,
            caller: sender,
            address: ENTRY_POINT,
            kind: CallKind::Call,
            gas_limit: 100,
            frame_transaction_root: true,
            ..Default::default()
        };

        let first = inspector.traces.push_trace(
            0,
            PushTraceKind::PushAndAttachToParent,
            frame_node(addresses[0], 0),
        );
        let failed = inspector.traces.push_trace(
            0,
            PushTraceKind::PushAndAttachToParent,
            CallTrace {
                status: Some(revm::interpreter::InstructionResult::Revert),
                ..frame_node(addresses[1], 1)
            },
        );
        let last = inspector.traces.push_trace(
            0,
            PushTraceKind::PushAndAttachToParent,
            frame_node(addresses[3], 3),
        );
        inspector.frame_transaction = Some(FrameTransactionTrace {
            root: 0,
            sender,
            frames,
            frame_nodes: vec![Some(first), Some(failed), None, Some(last)],
            pending_frame: None,
        });

        let result = ExecutionResult::<()>::FrameTransaction {
            gas: ResultGas::default().with_total_gas_spent(100),
            payer: sender,
            logs: Vec::new(),
            frame_receipts: vec![
                receipt(FrameStatus::Success, 4, 2),
                receipt(FrameStatus::Failure, 5, 3),
                receipt(FrameStatus::SkippedAtomicBatch, 0, 0),
                receipt(FrameStatus::Success, 6, 4),
            ],
            frame_outputs: vec![
                Bytes::from_static(&[1]),
                Bytes::from_static(&[2]),
                Bytes::new(),
                Bytes::from_static(&[4]),
            ],
        };

        inspector.finalize_frame_transaction(&result).unwrap();
        let root = &inspector.traces.arena[0];
        assert_eq!(root.trace.gas_used, 100);
        assert_eq!(root.children.len(), 4);
        assert_eq!(
            root.children
                .iter()
                .map(|node| inspector.traces.arena[*node].trace.frame_index)
                .collect::<Vec<_>>(),
            vec![Some(0), Some(1), Some(2), Some(3)]
        );

        let skipped = &inspector.traces.arena[root.children[2]].trace;
        assert_eq!(skipped.error.as_deref(), Some("frame skipped"));
        assert_eq!(skipped.gas_used, 0);
        assert!(skipped.output.is_empty());

        let traces = inspector.clone().into_parity_builder().into_transaction_traces();
        assert_eq!(traces.len(), 5);
        assert_eq!(traces[3].trace_address, vec![2]);
        assert_eq!(traces[3].error.as_deref(), Some("frame skipped"));
        assert!(traces[3].result.is_none());

        let calls =
            inspector.clone().into_geth_builder().geth_call_traces(CallConfig::default(), 100);
        assert_eq!(calls.calls.len(), 4);
        assert_eq!(calls.calls[2].error.as_deref(), Some("frame skipped"));
        assert_eq!(calls.calls[2].gas, U256::from(20));
        assert_eq!(calls.calls[2].gas_used, U256::ZERO);

        let vm_trace = inspector.into_parity_builder().vm_trace();
        assert_eq!(vm_trace.ops.len(), 3);
        assert_eq!(vm_trace.ops[0].pc, 0);
        assert_eq!(vm_trace.ops[1].pc, 1);
        assert_eq!(vm_trace.ops[2].pc, 3);
        assert_eq!(vm_trace.ops[0].ex.as_ref().unwrap().used, 14);
    }
}
