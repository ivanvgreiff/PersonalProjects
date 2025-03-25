from web3 import Web3
import requests

# Connect to an Ethereum node (I use Infura)
infura_url = "https://mainnet.infura.io/v3/8a8883700ab64f089d4512d70cff4e5c" 
web3 = Web3(Web3.HTTPProvider(infura_url))

# Query The Graph for DAO participation data
def fetch_dao_participation():
    graph_url = "https://gateway.thegraph.com/api/subgraphs/id/5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"

    # This is an old query vector, unfortunately it does not exist as such in the subgraph
    query = """
    {
        proposals(first: 10, orderBy: createdAt, orderDirection: desc) {
            id
            totalVotes
            quorum
            totalSupply
        }
    }
    """
    # Here I input my authorization API key so that I can query the subgraph
    headers = {"Authorization": "fa13f959a1838781762cf7f604973ddd"}
    response = requests.post(graph_url, json={"query": query}, headers=headers)
    print(response.json())

    if response.status_code == 200:
        data = response.json()
        daos = data["data"]["proposals"]
        for dao in daos:
            participation_rate = (float(dao["totalVotes"]) / float(dao["totalSupply"])) * 100
            print(f"DAO Proposal {dao['id']} - Participation Rate: {participation_rate:.2f}%")
    else:
        print("Failed to fetch data from The Graph.")

# Execute the function
fetch_dao_participation()
