import asyncio
import sys
import os

# Append app source directory to sys.path so we can import services
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

from services.graph_sync import GraphSyncService

async def main():
    print("======================================================================")
    print("GRAPH SYNC CLI INITIALIZATION: POSTGRESQL -> NEO4J")
    print("======================================================================")
    
    try:
        metrics = await GraphSyncService.sync_all_entities()
        print("✅ Graph database sync executed successfully!")
        print(f"   - Companies Synced: {metrics['companies']}")
        print(f"   - Jobs Synced:      {metrics['jobs']}")
        print(f"   - News Synced:      {metrics['news']}")
        print(f"   - Vessels Synced:   {metrics['vessels']}")
        print(f"   - Relationships:    {metrics['relationships']}")
        print("======================================================================")
    except Exception as e:
        print(f"❌ Graph sync failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
