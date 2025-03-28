from web3 import Web3
import json
from tqdm import tqdm

# Connect to Ethereum
web3 = Web3(Web3.HTTPProvider("https://mainnet.infura.io/v3/8a8883700ab64f089d4512d70cff4e5c"))

# Aave Governance Contract
GOVERNANCE_ADDRESS = "0xEC568fffba86c094cf06b22134B23074DFE2252c"
GOVERNANCE_ABI = json.loads('''[{"inputs":[],"name":"getProposalsCount","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint256","name":"proposalId","type":"uint256"}],"name":"getProposalById","outputs":[{"components":[{"internalType":"uint256","name":"id","type":"uint256"},{"internalType":"address","name":"creator","type":"address"},{"internalType":"contract IExecutorWithTimelock","name":"executor","type":"address"},{"internalType":"address[]","name":"targets","type":"address[]"},{"internalType":"uint256[]","name":"values","type":"uint256[]"},{"internalType":"string[]","name":"signatures","type":"string[]"},{"internalType":"bytes[]","name":"calldatas","type":"bytes[]"},{"internalType":"bool[]","name":"withDelegatecalls","type":"bool[]"},{"internalType":"uint256","name":"startBlock","type":"uint256"},{"internalType":"uint256","name":"endBlock","type":"uint256"},{"internalType":"uint256","name":"executionTime","type":"uint256"},{"internalType":"uint256","name":"forVotes","type":"uint256"},{"internalType":"uint256","name":"againstVotes","type":"uint256"},{"internalType":"bool","name":"executed","type":"bool"},{"internalType":"bool","name":"canceled","type":"bool"},{"internalType":"address","name":"strategy","type":"address"},{"internalType":"bytes32","name":"ipfsHash","type":"bytes32"}],"internalType":"struct IAaveGovernanceV2.ProposalWithoutVotes","name":"","type":"tuple"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint256","name":"proposalId","type":"uint256"},{"internalType":"address","name":"voter","type":"address"}],"name":"getVoteOnProposal","outputs":[{"components":[{"internalType":"bool","name":"support","type":"bool"},{"internalType":"uint248","name":"votingPower","type":"uint248"}],"internalType":"struct IAaveGovernanceV2.Vote","name":"","type":"tuple"}],"stateMutability":"view","type":"function"}]''')

governance = web3.eth.contract(address=GOVERNANCE_ADDRESS, abi=GOVERNANCE_ABI)

# Target address
TARGET_ADDRESS = "0xECC2a9240268BC7a26386ecB49E1Befca2706AC9"

def find_minority_votes():
    total_proposals = governance.functions.getProposalsCount().call()
    minority_votes = []
    
    print(f"Scanning {total_proposals} proposals...")
    
    for prop_id in tqdm(range(1, total_proposals + 1)):
        try:
            # Get proposal details
            proposal = governance.functions.getProposalById(prop_id).call()
            for_votes = proposal[10]  # Index 10 is forVotes
            against_votes = proposal[11]  # Index 11 is againstVotes
            
            # Get target address vote
            target_vote = governance.functions.getVoteOnProposal(prop_id, TARGET_ADDRESS).call()
            
            if target_vote:  # If the address voted on this proposal
                target_support = target_vote[0]
                target_power = target_vote[1]
                
                # Determine majority direction
                majority_support = for_votes > against_votes
                
                # Check if target voted against majority
                if target_support != majority_support:
                    minority_votes.append({
                        'proposal_id': prop_id,
                        'ipfs_hash': proposal[16].hex(),
                        'target_vote': 'FOR' if target_support else 'AGAINST',
                        'target_power': target_power,
                        'majority_direction': 'FOR' if majority_support else 'AGAINST',
                        'for_votes': for_votes,
                        'against_votes': against_votes
                    })
        except:
            continue
    
    return minority_votes

def print_results(results):
    print(f"\nFound {len(results)} proposals where {TARGET_ADDRESS} voted against the majority:")
    print("="*80)
    
    for result in results:
        print(f"\nProposal ID: {result['proposal_id']}")
        print(f"IPFS Hash: {result['ipfs_hash']}")
        print(f"\nTarget Vote: {result['target_vote']} (Power: {result['target_power']})")
        print(f"Majority Vote: {result['majority_direction']}")
        print(f"Total FOR Votes: {result['for_votes']}")
        print(f"Total AGAINST Votes: {result['against_votes']}")
        print(f"\nView on Boardroom: https://boardroom.io/aave/proposal/{result['ipfs_hash']}")
        print("="*80)

# Execute
results = find_minority_votes()
print_results(results)