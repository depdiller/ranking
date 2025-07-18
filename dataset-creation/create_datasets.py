import asyncio
from typing import Dict, List, Tuple

import pandas as pd
import datetime
import re
from pathlib import Path
import sys

from settings import Settings, load_mcp_servers_config
from mulit_mcp_facade import MCPFacade

sys.path.append(str(Path(__file__).parent.parent))

TOOL_TEST_QUERIES: Dict[str, List[Tuple[str, str]]] = {
    "InstanceServiceCreate": [
        ("english", "create a virtual machine instance"),
        ("english", "create instance"),
        ("english", "vm create"),
        ("english", "new virtual machine"),
        ("english", "launch instance"),
        ("english", "start new vm"),
        ("english", "provision instance"),
        ("english", "deploy virtual machine"),
        ("english", "create compute instance"),
        ("english", "new server instance"),
        ("english", "make virtual machine"),
        ("english", "build instance"),
        ("english", "setup vm"),
        ("english", "initialize instance"),
        ("russian", "создать виртуальную машину")
    ],
    "InstanceServiceDelete": [
        ("english", "delete virtual machine"),
        ("english", "remove instance"),
        ("english", "destroy vm"),
        ("english", "terminate instance"),
        ("english", "delete vm"),
        ("english", "remove virtual machine"),
        ("english", "destroy instance"),
        ("english", "kill vm"),
        ("english", "stop and delete instance"),
        ("english", "remove server"),
        ("english", "delete compute instance"),
        ("english", "terminate vm"),
        ("russian", "удалить виртуальную машину")
    ],
    "InstanceServiceStart": [
        ("english", "start instance"),
        ("english", "start virtual machine"),
        ("english", "start vm"),
        ("english", "power on instance"),
        ("english", "boot instance"),
        ("english", "turn on vm"),
        ("english", "activate instance"),
        ("english", "run instance"),
        ("english", "launch stopped vm"),
        ("english", "power up instance"),
        ("russian", "запустить экземпляр"),
        ("russian", "запустить виртуальную машину")
    ],
    "InstanceServiceStop": [
        ("english", "stop instance"),
        ("english", "stop virtual machine"),
        ("english", "stop vm"),
        ("english", "shutdown instance"),
        ("english", "power off vm"),
        ("english", "halt instance"),
        ("english", "turn off vm"),
        ("english", "pause instance"),
        ("english", "suspend vm"),
        ("english", "shutdown virtual machine"),
        ("russian", "остановить экземпляр"),
        ("russian", "остановить виртуальную машину")
    ],
    "InstanceServiceRestart": [
        ("english", "restart instance"),
        ("english", "restart virtual machine"),
        ("english", "restart vm"),
        ("english", "reboot instance"),
        ("english", "reboot vm"),
        ("english", "reset instance"),
        ("english", "cycle vm"),
        ("english", "restart server"),
        ("english", "reboot virtual machine"),
        ("english", "reset vm"),
        ("russian", "перезапустить экземпляр"),
        ("russian", "перезагрузить виртуальную машину")
    ],
    "InstanceServiceUpdate": [
        ("english", "update instance"),
        ("english", "modify instance"),
        ("english", "change instance"),
        ("english", "edit vm"),
        ("english", "update virtual machine"),
        ("english", "modify vm configuration"),
        ("english", "change vm settings"),
        ("english", "update instance config"),
        ("english", "edit instance properties"),
        ("english", "modify instance settings"),
        ("russian", "обновить экземпляр"),
        ("russian", "изменить экземпляр")
    ],
    "InstanceServiceList": [
        ("english", "list instances"),
        ("english", "show instances"),
        ("english", "get instances"),
        ("english", "list virtual machines"),
        ("english", "show vms"),
        ("english", "list all instances"),
        ("english", "get vm list"),
        ("english", "show all virtual machines"),
        ("english", "enumerate instances"),
        ("english", "display instances"),
        ("russian", "список экземпляров"),
        ("russian", "показать экземпляры")
    ],
    "InstanceServiceGet": [
        ("english", "get instance"),
        ("english", "show instance"),
        ("english", "describe instance"),
        ("english", "get vm details"),
        ("english", "show virtual machine"),
        ("english", "instance details"),
        ("english", "vm information"),
        ("english", "get instance info"),
        ("english", "describe vm"),
        ("english", "show instance details"),
        ("russian", "получить экземпляр"),
        ("russian", "показать экземпляр")
    ],
    "DiskServiceCreate": [
        ("english", "create disk"),
        ("english", "create storage"),
        ("english", "new disk"),
        ("english", "add disk"),
        ("english", "provision storage"),
        ("english", "create volume"),
        ("english", "make disk"),
        ("english", "setup storage"),
        ("english", "create block storage"),
        ("english", "new volume"),
        ("english", "add storage disk"),
        ("english", "provision disk"),
        ("russian", "создать диск"),
        ("russian", "создать хранилище")
    ],
    "DiskServiceDelete": [
        ("english", "delete disk"),
        ("english", "remove disk"),
        ("english", "destroy storage"),
        ("english", "delete volume"),
        ("english", "remove storage"),
        ("english", "destroy disk"),
        ("english", "delete block storage"),
        ("english", "remove volume"),
        ("english", "destroy volume"),
        ("english", "delete storage disk"),
        ("russian", "удалить диск"),
        ("russian", "удалить хранилище")
    ],
    "DiskServiceUpdate": [
        ("english", "update disk"),
        ("english", "modify disk"),
        ("english", "change disk"),
        ("english", "edit storage"),
        ("english", "update volume"),
        ("english", "modify storage"),
        ("english", "change volume"),
        ("english", "update disk settings"),
        ("english", "modify disk configuration"),
        ("english", "edit disk properties"),
        ("russian", "обновить диск"),
        ("russian", "изменить диск")
    ],
    "DiskServiceList": [
        ("english", "list disks"),
        ("english", "show disks"),
        ("english", "get disks"),
        ("english", "list storage"),
        ("english", "show volumes"),
        ("english", "list all disks"),
        ("english", "get disk list"),
        ("english", "show all storage"),
        ("english", "enumerate disks"),
        ("english", "display disks"),
        ("russian", "список дисков"),
        ("russian", "показать диски")
    ],
    "SnapshotServiceCreate": [
        ("english", "create snapshot"),
        ("english", "backup disk"),
        ("english", "take snapshot"),
        ("english", "create backup"),
        ("english", "snapshot disk"),
        ("english", "make snapshot"),
        ("english", "backup volume"),
        ("english", "create disk snapshot"),
        ("english", "take disk backup"),
        ("english", "snapshot storage"),
        ("english", "backup storage"),
        ("english", "create volume snapshot"),
        ("russian", "создать снимок"),
        ("russian", "создать резервную копию")
    ],
    "SnapshotServiceDelete": [
        ("english", "delete snapshot"),
        ("english", "remove snapshot"),
        ("english", "destroy snapshot"),
        ("english", "delete backup"),
        ("english", "remove backup"),
        ("english", "destroy backup"),
        ("english", "delete disk snapshot"),
        ("english", "remove disk backup"),
        ("english", "destroy disk snapshot"),
        ("english", "delete volume snapshot"),
        ("russian", "удалить снимок"),
        ("russian", "удалить резервную копию")
    ],
    "SnapshotServiceList": [
        ("english", "list snapshots"),
        ("english", "show snapshots"),
        ("english", "get snapshots"),
        ("english", "list backups"),
        ("english", "show backups"),
        ("english", "list all snapshots"),
        ("english", "get snapshot list"),
        ("english", "show all backups"),
        ("english", "enumerate snapshots"),
        ("english", "display snapshots"),
        ("russian", "список снимков"),
        ("russian", "показать снимки")
    ],
    "ImageServiceCreate": [
        ("english", "create image"),
        ("english", "create template"),
        ("english", "build image"),
        ("english", "make image"),
        ("english", "create vm image"),
        ("english", "create disk image"),
        ("english", "build template"),
        ("english", "make template"),
        ("english", "create system image"),
        ("english", "build vm template"),
        ("english", "create machine image"),
        ("english", "make disk image"),
        ("russian", "создать образ"),
        ("russian", "создать шаблон")
    ],
    "ImageServiceDelete": [
        ("english", "delete image"),
        ("english", "remove image"),
        ("english", "destroy image"),
        ("english", "delete template"),
        ("english", "remove template"),
        ("english", "destroy template"),
        ("english", "delete vm image"),
        ("english", "remove disk image"),
        ("english", "destroy machine image"),
        ("english", "delete system image"),
        ("russian", "удалить образ"),
        ("russian", "удалить шаблон")
    ],
    "ImageServiceList": [
        ("english", "list images"),
        ("english", "show images"),
        ("english", "get images"),
        ("english", "list templates"),
        ("english", "show templates"),
        ("english", "list all images"),
        ("english", "get image list"),
        ("english", "show all templates"),
        ("english", "enumerate images"),
        ("english", "display images"),
        ("russian", "список образов"),
        ("russian", "показать образы")
    ],
    "NetworkServiceCreate": [
        ("english", "create network"),
        ("english", "create vpc"),
        ("english", "new network"),
        ("english", "setup network"),
        ("english", "provision network"),
        ("english", "make network"),
        ("english", "create virtual network"),
        ("english", "setup vpc"),
        ("english", "create private network"),
        ("english", "build network"),
        ("english", "establish network"),
        ("english", "create subnet"),
        ("russian", "создать сеть"),
        ("russian", "создать VPC")
    ],
    "NetworkServiceDelete": [
        ("english", "delete network"),
        ("english", "remove network"),
        ("english", "destroy network"),
        ("english", "delete vpc"),
        ("english", "remove vpc"),
        ("english", "destroy vpc"),
        ("english", "delete virtual network"),
        ("english", "remove private network"),
        ("english", "destroy virtual network"),
        ("english", "delete subnet"),
        ("russian", "удалить сеть"),
        ("russian", "удалить VPC")
    ],
    "SubnetServiceCreate": [
        ("english", "create subnet"),
        ("english", "create subnetwork"),
        ("english", "new subnet"),
        ("english", "setup subnet"),
        ("english", "provision subnet"),
        ("english", "make subnet"),
        ("english", "create network subnet"),
        ("english", "build subnet"),
        ("english", "establish subnet"),
        ("english", "create vpc subnet"),
        ("russian", "создать подсеть"),
        ("russian", "создать субсеть")
    ],
    "SubnetServiceDelete": [
        ("english", "delete subnet"),
        ("english", "remove subnet"),
        ("english", "destroy subnet"),
        ("english", "delete subnetwork"),
        ("english", "remove subnetwork"),
        ("english", "destroy subnetwork"),
        ("english", "delete network subnet"),
        ("english", "remove vpc subnet"),
        ("english", "destroy vpc subnet"),
        ("english", "delete private subnet"),
        ("russian", "удалить подсеть"),
        ("russian", "удалить субсеть")
    ],
    "ZoneServiceList": [
        ("english", "list zones"),
        ("english", "get availability zones"),
        ("english", "show zones"),
        ("english", "list regions"),
        ("english", "available zones"),
        ("english", "show availability zones"),
        ("english", "get zones"),
        ("english", "list all zones"),
        ("english", "show regions"),
        ("english", "enumerate zones"),
        ("english", "display zones"),
        ("english", "get region list"),
        ("russian", "список зон"),
        ("russian", "показать зоны")
    ],
    "ZoneServiceGet": [
        ("english", "get zone"),
        ("english", "show zone"),
        ("english", "describe zone"),
        ("english", "get availability zone"),
        ("english", "show availability zone"),
        ("english", "zone details"),
        ("english", "zone information"),
        ("english", "get zone info"),
        ("english", "describe availability zone"),
        ("english", "show zone details"),
        ("russian", "получить зону"),
        ("russian", "показать зону")
    ],
    "GpuClusterServiceCreate": [
        ("english", "create gpu cluster"),
        ("english", "create gpu"),
        ("english", "new gpu cluster"),
        ("english", "setup gpu cluster"),
        ("english", "provision gpu"),
        ("english", "make gpu cluster"),
        ("english", "create graphics cluster"),
        ("english", "setup gpu farm"),
        ("english", "create compute cluster"),
        ("english", "build gpu cluster"),
        ("english", "establish gpu cluster"),
        ("english", "create gpu pool"),
        ("russian", "создать GPU кластер"),
        ("russian", "создать графический кластер")
    ],
    "GpuClusterServiceDelete": [
        ("english", "delete gpu cluster"),
        ("english", "remove gpu cluster"),
        ("english", "destroy gpu cluster"),
        ("english", "delete gpu"),
        ("english", "remove gpu"),
        ("english", "destroy gpu"),
        ("english", "delete graphics cluster"),
        ("english", "remove compute cluster"),
        ("english", "destroy gpu farm"),
        ("english", "delete gpu pool"),
        ("russian", "удалить GPU кластер"),
        ("russian", "удалить графический кластер")
    ],
    "FilesystemServiceCreate": [
        ("english", "create filesystem"),
        ("english", "create file system"),
        ("english", "new filesystem"),
        ("english", "setup filesystem"),
        ("english", "provision filesystem"),
        ("english", "make filesystem"),
        ("english", "create shared storage"),
        ("english", "setup file storage"),
        ("english", "create network filesystem"),
        ("english", "build filesystem"),
        ("english", "establish filesystem"),
        ("english", "create shared filesystem"),
        ("russian", "создать файловую систему"),
        ("russian", "создать файловое хранилище")
    ],
    "FilesystemServiceDelete": [
        ("english", "delete filesystem"),
        ("english", "remove filesystem"),
        ("english", "destroy filesystem"),
        ("english", "delete file system"),
        ("english", "remove file system"),
        ("english", "destroy file system"),
        ("english", "delete shared storage"),
        ("english", "remove file storage"),
        ("english", "destroy network filesystem"),
        ("english", "delete shared filesystem"),
        ("russian", "удалить файловую систему"),
        ("russian", "удалить файловое хранилище")
    ]
}


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
            raise e
            # continue

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
            resource, method = _parse_tool_name(tool_name)
            generation_gt = resource + method
            
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
    
    # # Convert retrieval_gt to proper format before saving
    # # AutoRAG expects 2D list format, but parquet may convert to numpy arrays
    # qa_df['retrieval_gt'] = qa_df['retrieval_gt'].apply(lambda x: x if isinstance(x, list) else x.tolist())
    
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
