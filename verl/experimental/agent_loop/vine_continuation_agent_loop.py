"""Generate scored full responses from token-exact intermediate prefixes."""

from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.utils.profiler import simple_timer
from verl.workers.rollout.replica import TokenOutput


@register("vine_continuation_agent")
class VineContinuationAgentLoop(SingleTurnAgentLoop):
    """Use a prefix for sampling while exposing the full answer to reward code."""

    async def run(self, sampling_params: dict, **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        multi_modal_data = await self.process_vision_info(messages)
        if multi_modal_data.get("images") or multi_modal_data.get("videos"):
            raise ValueError("Vine continuation currently supports text-only prompts")
        prefix_ids = [int(token) for token in kwargs["vine_prefix_ids"]]
        if len(prefix_ids) >= self.response_length:
            raise ValueError("Vine continuation needs a nonterminal response prefix")
        prompt_ids = await self.apply_chat_template(messages)
        remaining = self.response_length - len(prefix_ids)
        params = dict(sampling_params)
        params["max_tokens"] = remaining
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=[*prompt_ids, *prefix_ids],
                sampling_params=params,
            )
        metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        response_ids = [*prefix_ids, *output.token_ids[:remaining]]
        result = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=[1] * len(response_ids),
            response_logprobs=None,
            routed_experts=None,
            multi_modal_data={},
            num_turns=2,
            metrics=metrics,
            extra_fields=output.extra_fields,
        )
        result.extra_fields.update({"turn_scores": [], "tool_rewards": []})
        return result
