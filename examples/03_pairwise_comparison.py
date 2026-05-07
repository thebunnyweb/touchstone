"""Pairwise comparison with position-bias correction."""
import asyncio
from touchstone import Judge

async def main():
    judge = Judge(model="claude-sonnet-4-6")

    output_a = "Python is a high-level, interpreted programming language known for its readability."
    output_b = "Python is a programming language."

    result = await judge.compare(output_a, output_b, rubric=["completeness", "clarity"])

    print(f"Winner:                {result.winner}")
    print(f"Confidence:            {result.confidence:.2f}")
    print(f"Score A:               {result.score_a:.3f}")
    print(f"Score B:               {result.score_b:.3f}")
    print(f"Position bias flagged: {result.position_bias_detected}")
    print(f"Reasoning:             {result.reasoning}")

if __name__ == "__main__":
    asyncio.run(main())
