import asyncio
import pandas as pd
import datetime
import re
from pathlib import Path
import sys

from utils.settings import Settings, load_mcp_servers_config
from utils.mulit_mcp_facade import MCPFacade

sys.path.append(str(Path(__file__).parent.parent))

from tests.fixtures.test_data import TOOL_TEST_QUERIES

async def create_corpus_dataset():
    """Create corpus dataset with tool documentation following AutoRAG format using real MCP tools."""
    settings = Settings()
    mcp_servers_config = load_mcp_servers_config(settings.mcp_servers_file)
    mcp_facade = MCPFacade(settings=settings)
    
    corpus_data = []
    
    # Connect to MCP servers and get real tool data (same as in rag.py __ingest_tools)
    for server_id, _ in mcp_servers_config.items():
        try:
            print(f"Fetching tools from MCP server {server_id}")
            tools = await mcp_facade.list_tools(server_id)

            if not tools:
                print(f"No tools found in server {server_id}")
                continue

            for tool in tools:
                tool_id = f"{server_id}.{tool.name}"
                description = tool.description or ""

                # Parse tool name to get resource and method (same as in rag.py)
                resource, method = _parse_tool_name(tool.name)

                # Create cleaned name and description (same as in rag.py)
                cleaned_name = re.sub(r"service", "", tool.name, flags=re.IGNORECASE)
                cleaned_description = re.sub(r"service", "", description, flags=re.IGNORECASE)

                # Create contents exactly like in rag.py
                doc_content = f"{cleaned_name} - {cleaned_description}"

                # Create metadata similar to rag.py but with required last_modified_datetime
                metadata = {
                    "tool_name": tool.name,
                    "resource": resource,
                    "method": method,
                    "service": server_id,
                    "last_modified_datetime": datetime.datetime.now()
                }

                corpus_data.append({
                    "doc_id": tool_id,  # Use tool_id like in rag.py
                    "contents": doc_content,
                    "metadata": metadata
                })

            print(f"Loaded {len(tools)} tools from {server_id}")
        except FileNotFoundError:
            print(f"Skipping {server_id} - MCP server executable not found")
            continue
        except Exception as e:
            print(f"Failed to fetch tools from {server_id}: {e}")
            continue

    corpus_df = pd.DataFrame(corpus_data)
    return corpus_df

def _parse_tool_name(tool_name: str) -> tuple[str, str]:
    """
    Parse tool_name to extract resource and method.
    Format: {Resource}Service{Method}
    Resource: everything before 'Service'
    Method: everything after 'Service'
    Same as in rag.py
    """
    if not tool_name:
        return "", ""

    service_index = tool_name.find("Service")

    if service_index == -1:
        return tool_name, ""

    resource = tool_name[:service_index]
    method = tool_name[service_index + 7:]

    return resource, method

async def create_qa_dataset(corpus_df):
    """Create QA dataset with queries and ground truth using real tool IDs from corpus."""
    qa_data = []
    
    # Get mapping from tool names to actual doc_ids from corpus
    tool_name_to_doc_id = {}
    for _, row in corpus_df.iterrows():
        tool_name = row['metadata']['tool_name']
        doc_id = row['doc_id']
        tool_name_to_doc_id[tool_name] = doc_id
    
    for tool_name, queries in TOOL_TEST_QUERIES.items():
        # Skip if tool not found in actual MCP servers
        if tool_name not in tool_name_to_doc_id:
            continue
            
        doc_id = tool_name_to_doc_id[tool_name]
        
        for lang, query in queries:
            # Create unique query ID
            qid = f"{tool_name}_{lang}_{len(qa_data)}"
            
            # Retrieval ground truth - 2D list of document IDs (as required by AutoRAG)
            retrieval_gt = [[doc_id]]  # Use actual doc_id from corpus
            
            # Generation ground truth - just the tool name
            generation_gt = tool_name
            
            qa_data.append({
                "qid": qid,
                "query": query,
                "retrieval_gt": retrieval_gt,
                "generation_gt": generation_gt
            })
    
    qa_df = pd.DataFrame(qa_data)
    return qa_df

async def main():
    """Create and save datasets."""
    # Create datasets
    print("Creating corpus dataset from MCP tools...")
    corpus_df = await create_corpus_dataset()
    
    print("Creating QA dataset...")
    qa_df = await create_qa_dataset(corpus_df)
    
    # Ensure autorag directory exists
    autorag_dir = Path("../data")
    autorag_dir.mkdir(exist_ok=True)
    
    # Convert retrieval_gt to proper format before saving
    # AutoRAG expects 2D list format, but parquet may convert to numpy arrays
    qa_df['retrieval_gt'] = qa_df['retrieval_gt'].apply(lambda x: x if isinstance(x, list) else x.tolist())
    
    # Save datasets
    corpus_df.to_parquet(autorag_dir / "corpus.parquet", index=False)
    qa_df.to_parquet(autorag_dir / "qa.parquet", index=False)
    
    print(f"Created corpus dataset with {len(corpus_df)} documents")
    print(f"Created QA dataset with {len(qa_df)} queries")
    print(f"Datasets saved to {autorag_dir}")
    
    # Print sample data
    print("\nSample corpus data:")
    print(corpus_df.head(2))
    print("\nSample QA data:")
    print(qa_df.head(5))
    
    # Verify format after saving and loading
    print("\nVerifying saved format:")
    saved_qa_df = pd.read_parquet(autorag_dir / "qa.parquet")
    saved_corpus_df = pd.read_parquet(autorag_dir / "corpus.parquet")
    
    print(f"Retrieval_gt type: {type(saved_qa_df.iloc[0]['retrieval_gt'])}")
    print(f"Retrieval_gt sample: {saved_qa_df.iloc[0]['retrieval_gt']}")
    print(f"Metadata type: {type(saved_corpus_df.iloc[0]['metadata'])}")
    print(f"Metadata keys: {list(saved_corpus_df.iloc[0]['metadata'].keys())}")

if __name__ == "__main__":
    asyncio.run(main())
