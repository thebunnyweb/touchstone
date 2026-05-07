"""Basic single-judge scoring example."""
import asyncio
from touchstone import Judge, Rubric

async def main():
    judge = Judge(model="claude-sonnet-4-6")

    output = "The Eiffel Tower is located in Paris, France. It was built in 1889."

    rubric = (
        Rubric()
        .add("factual accuracy", "All facts are correct and verifiable", weight=0.6)
        .add("conciseness", "No unnecessary padding", weight=0.4)
    )

    result = await judge.score(output, rubric=rubric)

    print(f"Score:          {result.score:.3f}")
    print(f"Adjusted score: {result.adjusted_score:.3f}")
    print(f"Confidence:     {result.confidence:.2f}")
    print(f"Passed:         {result.passed}")
    print(f"Bias flags:     {result.bias_flags or 'none'}")
    print(f"Reasoning:      {result.reasoning}")
    print()
    for cs in result.criteria_scores:
        print(f"  {cs.name}: {cs.score:.2f} — {cs.reasoning}")

if __name__ == "__main__":
    asyncio.run(main())
