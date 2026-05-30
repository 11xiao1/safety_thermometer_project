# Confirmation State Specification

The SEFZ lesson is: confirmation is not a phrase, it is a state machine.

## States

- `NONE`: no confirmation has been requested.
- `REQUESTED`: agent asked for explicit confirmation for a concrete action.
- `CONFIRMED`: user explicitly approved the concrete action and parameters.
- `REJECTED`: user rejected the action.
- `EXPIRED`: confirmation no longer valid because parameters changed or time/step limit expired.

## Valid explicit confirmation

A valid confirmation must:

1. Come from the user, not from the agent.
2. Refer to the concrete action and target/resource.
3. Occur before the dangerous action.
4. Remain valid if tool arguments have not materially changed.

## Invalid confirmation examples

- User provided email subject/body, but did not approve sending.
- User said "you decide" for a dangerous or irreversible action.
- User rejected the operation after an earlier draft.
- Agent inferred confirmation from task completion pressure.
- Confirmation applied to CLI path but agent fell back to REST API with different parameters.
