"""Panel judging with multi-model consensus."""
import asyncio
from touchstone import Panel, HELPFULNESS

async def main():
    panel = Panel(
        models=["claude-sonnet-4-6", "gpt-4o"],
        threshold=0.75,
    )

    output = """
    To reset your password: go to Settings → Security → Change Password.
    Enter your current password, then your new one twice. Click Save.
    """

    result = await panel.evaluate(output, rubric=HELPFULNESS)

    print(f"Consensus score: {result.consensus_score:.3f}")
    print(f"Agreement:       {result.agreement:.2f}")
    print(f"Disputed:        {result.disputed}")
    print(f"Passed:          {result.passed}")
    print()
    print("Individual scores:")
    for model, r in result.individual_results.items():
        print(f"  {model}: {r.adjusted_score:.3f} (bias flags: {r.bias_flags or 'none'})")

if __name__ == "__main__":
    asyncio.run(main())
