use alloy_eip8141::{Frame, FrameAddress, FrameLimits, FrameMode, FrameStatus};
use alloy_primitives::{address, Bytes, TxKind, U256};
use alloy_rpc_types_trace::geth::CallConfig;
use revm::{
    bytecode::{opcode::STOP, Bytecode},
    context::{transaction::FrameTransaction, TxEnv},
    context_interface::result::ExecutionResult,
    database::CacheDB,
    database_interface::EmptyDB,
    primitives::hardfork::SpecId,
    state::AccountInfo,
    Context, InspectEvm, MainBuilder, MainContext,
};
use revm_inspectors::tracing::TracingInspector;

const SENDER: alloy_primitives::Address = address!("1000000000000000000000000000000000000001");
const FAILING_TARGET: alloy_primitives::Address =
    address!("2000000000000000000000000000000000000002");
const SUCCESS_TARGET: alloy_primitives::Address =
    address!("3000000000000000000000000000000000000003");

fn account_with_code(code: impl Into<Bytes>) -> AccountInfo {
    AccountInfo::default().with_code(Bytecode::new_legacy(code.into()))
}

fn tx_env(payload: FrameTransaction) -> TxEnv {
    let gas_limit = payload.gas_limit(SENDER).unwrap();
    TxEnv::builder()
        .tx_type(Some(0x06))
        .caller(SENDER)
        .kind(TxKind::Call(SENDER))
        .gas_limit(gas_limit)
        .gas_priority_fee(Some(0))
        .frame_transaction(payload)
        .build()
        .unwrap()
}

fn trace_pre_evm_failure(execution_limit: u64, value: U256) {
    let mut db = CacheDB::<EmptyDB>::default();
    db.insert_account_info(SENDER, account_with_code([0x60, 0x03, 0x5f, 0x5f, 0xaa, STOP]));
    db.insert_account_info(FAILING_TARGET, account_with_code([STOP]));
    db.insert_account_info(SUCCESS_TARGET, account_with_code([STOP]));

    let frames = vec![
        Frame {
            flags: 0x03,
            limits: FrameLimits { execution: 10_000, state: 0 },
            ..Default::default()
        },
        Frame {
            mode: FrameMode::Sender,
            target: FrameAddress::from(FAILING_TARGET),
            limits: FrameLimits { execution: execution_limit, state: 0 },
            value,
            ..Default::default()
        },
        Frame {
            mode: FrameMode::Sender,
            target: FrameAddress::from(SUCCESS_TARGET),
            limits: FrameLimits { execution: 10_000, state: 0 },
            ..Default::default()
        },
    ];
    let payload = FrameTransaction { frames, ..Default::default() };
    let mut inspector = TracingInspector::default();
    let result = {
        let context = Context::mainnet()
            .modify_cfg_chained(|cfg| cfg.set_spec_and_mainnet_gas_params(SpecId::BOGOTA))
            .with_db(db);
        let mut evm = context.build_mainnet().with_inspector(&mut inspector);
        evm.inspect_tx(tx_env(payload)).unwrap().result
    };

    let ExecutionResult::FrameTransaction { ref frame_receipts, .. } = result else {
        panic!("expected frame transaction result, got {result:?}")
    };
    assert_eq!(frame_receipts[1].status, FrameStatus::Failure);
    inspector.finalize_frame_transaction(&result).unwrap();

    let calls = inspector
        .clone()
        .into_geth_builder()
        .geth_call_traces(CallConfig::default(), result.tx_gas_used());
    assert_eq!(calls.calls.len(), 3);
    assert_eq!(calls.calls[1].error.as_deref(), Some("frame failed"));
    assert_eq!(
        calls.calls[1].gas_used,
        U256::from(frame_receipts[1].gas_used.execution + frame_receipts[1].gas_used.state)
    );
    assert!(calls.calls[1].calls.is_empty());

    let parity = inspector.into_parity_builder().into_transaction_traces();
    let failed = parity.iter().find(|trace| trace.trace_address == [1]).unwrap();
    assert_eq!(failed.error.as_deref(), Some("frame failed"));
    assert!(failed.result.is_none());
    assert_eq!(failed.subtraces, 0);
}

#[test]
fn traces_frame_that_runs_out_of_gas_before_evm_entry() {
    trace_pre_evm_failure(1, U256::ZERO);
}

#[test]
fn traces_frame_that_lacks_balance_before_evm_entry() {
    trace_pre_evm_failure(10_000, U256::from(1));
}
