"""Calibration drift detection — detect silent model updates."""
import asyncio
from touchstone import Judge, Calibration, FACTUALITY

REFERENCE_SAMPLES = [
    "Water boils at 100°C at sea level.",
    "The speed of light is approximately 299,792 km/s.",
    "DNA is composed of four nucleotide bases: adenine, thymine, guanine, and cytosine.",
    "The Great Wall of China is approximately 21,196 km long.",
    "Photosynthesis converts sunlight, water, and CO2 into glucose and oxygen.",
]

async def main():
    judge = Judge(model="claude-sonnet-4-6")
    cal = Calibration(judge, store=".touchstone/calibration.json")

    print("Recording baseline...")
    await cal.baseline(REFERENCE_SAMPLES, rubric=FACTUALITY)

    print("\nChecking for drift...")
    report = await cal.check(REFERENCE_SAMPLES, rubric=FACTUALITY)

    print(report)
    if report.drifted:
        print("⚠ Model scoring behavior has shifted — review your evals.")
    else:
        print("✓ Judge is stable.")

if __name__ == "__main__":
    asyncio.run(main())
