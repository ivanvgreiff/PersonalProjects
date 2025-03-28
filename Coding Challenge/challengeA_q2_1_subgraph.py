from web3 import Web3
import requests

# Connect to an Ethereum node (I use Infura)
infura_url = "https://mainnet.infura.io/v3/8a8883700ab64f089d4512d70cff4e5c" 
web3 = Web3(Web3.HTTPProvider(infura_url))

# The two addresses we want to compare votes for
#ADDRESS1 = "0x8b37a5Af68D315cf5A64097D96621F64b5502a22"
#ADDRESS2 = "0xECC2a9240268BC7a26386ecB49E1Befca2706AC9"
ADDRESS1 = "0x8b37a5af68d315cf5a64097d96621f64b5502a22"  
ADDRESS2 = "0xecc2a9240268bc7a26386ecb49e1befca2706ac9"

def fetch_differing_votes():
    graph_url = "https://gateway.thegraph.com/api/subgraphs/id/8EBbn3tNayccBZrnW9ae6Q4NLHfVEcozvkB3YAp5Qatr"
    
    query = """
    {
        proposals(first: 1000, skip:1000, orderBy: creationTime, orderDirection: desc) {
            id
            votes(where: {voter_in: ["0x8b37a5af68d315cf5a64097d96621f64b5502a22", "0xecc2a9240268bc7a26386ecb49e1befca2706ac9"]}) {
                choice
                id
            }
        }
    }""" 
    
    headers = {"Authorization": "Bearer fa13f959a1838781762cf7f604973ddd"}
    response = requests.post(graph_url, json={"query": query}, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        proposals = data["data"]["proposals"]
        
        differing_proposals = []
        
        for proposal in proposals:
            votes = proposal["votes"]
            
            # Check if both addresses voted on this proposal
            voters = [vote["voter"] for vote in votes]
            if ADDRESS1.lower() in voters and ADDRESS2.lower() in voters:
                # Get their votes
                vote1 = next(vote for vote in votes if vote["voter"] == ADDRESS1.lower())
                vote2 = next(vote for vote in votes if vote["voter"] == ADDRESS2.lower())
                
                # Check if they voted differently (support is boolean: true/false)
                if vote1["support"] != vote2["support"]:
                    differing_proposals.append({
                        "proposal_id": proposal["id"],
                        "proposal_title": proposal.get("title", "No title"),
                        ADDRESS1: "For" if vote1["support"] else "Against",
                        ADDRESS2: "For" if vote2["support"] else "Against",
                        "voting_power1": vote1["votingPower"],
                        "voting_power2": vote2["votingPower"]
                    })
        
        # Print results
        print(f"Found {len(differing_proposals)} proposals where the addresses voted differently:")
        for prop in differing_proposals:
            print(f"\nProposal ID: {prop['proposal_id']}")
            print(f"Title: {prop['proposal_title']}")
            print(f"{ADDRESS1}: {prop[ADDRESS1]} (Voting Power: {prop['voting_power1']})")
            print(f"{ADDRESS2}: {prop[ADDRESS2]} (Voting Power: {prop['voting_power2']})")
            
        return differing_proposals
    else:
        print("Failed to fetch data from The Graph.")
        print("Response:", response.text)
        return None

# Execute the function
fetch_differing_votes()
