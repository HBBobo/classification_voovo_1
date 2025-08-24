from utils.active_learning_pipeline import run_pipeline_async
import os
import asyncio

async def main():
    """
    The project's central asynchronous entry point.
    """
    print("=============================================")
    print("  Document Classification - Active Learning  ")
    print("=============================================")
    
    if not os.getenv("GOOGLE_API_KEY"):
        print("\nError: no API key found. Please set the GOOGLE_API_KEY in your .env file.")
        return

    try:
        await run_pipeline_async()
    except Exception as e:
        print(f"\nAn error occurred during the async pipeline: {e}")

if __name__ == "__main__":
    asyncio.run(main())