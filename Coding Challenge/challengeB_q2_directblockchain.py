import requests
from web3 import Web3
import pandas as pd

# Configuration
INFURA_URL = "https://mainnet.infura.io/v3/8a8883700ab64f089d4512d70cff4e5c"
GOVERNANCE_ADDRESS = "0x9AEE0B04504CeF83A65AC3f0e838D0593BCb2BC7"  # Corrected address
HELPER_ADDRESS = "0x971c82c8316aD611904F95616c21ce90837f1856"    # Corrected helper address

# Minimal ABIs
GOVERNANCE_ABI = [{
    "inputs": [],
    "name": "getProposalsCount",
    "outputs": [{"name": "", "type": "uint256"}],
    "stateMutability": "view",
    "type": "function"
}]

HELPER_ABI = [{
    "inputs": [
        {"name": "govCore", "type": "address"},
        {"name": "from", "type": "uint256"},
        {"name": "to", "type": "uint256"},
        {"name": "pageSize", "type": "uint256"}
    ],
    "name": "getProposalsData",
    "outputs": [{
        "components": [
            {"name": "id", "type": "uint256"},
            {
                "components": [
                    {"name": "id", "type": "uint256"},
                    {"name": "state", "type": "uint8"},
                    {"name": "creator", "type": "address"},
                    {"name": "forVotes", "type": "uint128"},
                    {"name": "againstVotes", "type": "uint128"},
                ],
                "name": "proposalData",
                "type": "tuple"
            }
        ],
        "name": "",
        "type": "tuple[]"
    }],
    "stateMutability": "view",
    "type": "function"
}]

# Initialize Web3 with Infura
w3 = Web3(Web3.HTTPProvider(INFURA_URL))

# Initialize contracts with corrected addresses
governance = w3.eth.contract(address=GOVERNANCE_ADDRESS, abi=GOVERNANCE_ABI)
helper = w3.eth.contract(address=HELPER_ADDRESS, abi=HELPER_ABI)

def get_proposals(limit=10):
    """Get proposals with proper error handling"""
    try:
        # Get total count first
        total = governance.functions.getProposalsCount().call()
        print(f"Total proposals on chain: {total}")
        
        if total == 0:
            return []

        # Calculate range (handling potential underflow)
        from_id = max(1, total - limit + 1)
        
        # Get proposal data
        proposals = helper.functions.getProposalsData(
            GOVERNANCE_ADDRESS,  # Using the corrected governance address
            from_id,
            total,
            limit
        ).call()
        
        return proposals[0]  # Returns the tuple array

    except Exception as e:
        print(f"Error fetching proposals: {str(e)}")
        return []

def format_proposals(raw_proposals):
    """Format raw proposal data into readable format"""
    state_mapping = {
        0: "Pending", 1: "Canceled", 2: "Active",
        3: "Failed", 4: "Succeeded", 5: "Queued",
        6: "Expired", 7: "Executed"
    }
    
    formatted = []
    for prop in raw_proposals:
        prop_id = prop[0]
        data = prop[1]
        formatted.append({
            'ID': prop_id,
            'State': state_mapping.get(data[1], f"Unknown ({data[1]})"),
            'Creator': data[2],
            'For Votes': data[3],
            'Against Votes': data[4],
            'URL': f"https://app.aave.com/governance/proposal/{prop_id}"
        })
    
    return formatted

if __name__ == "__main__":
    print("Fetching AAVE governance proposals...")
    
    # Get raw proposal data
    raw_proposals = get_proposals(limit=15)  # Get last 15 proposals
    
    if raw_proposals:
        # Process and display
        df = pd.DataFrame(format_proposals(raw_proposals))
        
        print("\nLatest Proposals:")
        print(df[['ID', 'State', 'For Votes', 'Against Votes', 'URL']].to_markdown(index=False))
    else:
        print("No proposals found or error occurred")