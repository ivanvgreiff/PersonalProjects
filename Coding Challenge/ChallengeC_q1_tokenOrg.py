from web3 import Web3
import requests

"""
This Python script calculates the total unrealized revenue in units of [token-type]
It consider's Jared's bot's sandwich trades

The next file in this challenge calculates the total unrealized revenue in units of [USD]

Note: I only scan through 10,000 transactions as this is the maximum which Etherscan returns 
to us when using their free account. To scan through more transactions, we can simply create a while 
loop within the get_transactions() function and query multiple batches of blocks as we iterate
"""

# Connect to an Ethereum node - I am using my Infura key here
INFURA_URL = "https://mainnet.infura.io/v3/8a8883700ab64f089d4512d70cff4e5c"
web3 = Web3(Web3.HTTPProvider(INFURA_URL))

# Jared's (bot) wallet address
jared_address = "0x6b75d8af000000e20b7a7ddf000ba900b4009a80"

def get_transactions(address):
    # API Key to get access to Etherscan's free query responses
    etherscan_api_key = "YJQS43D2JKVXJZSX2EEQ587KDHC8MNSKQ6"

    # Transactions URL
    #url = f"https://api.etherscan.io/api?module=account&action=txlist&address={address}&startblock=0&endblock=99999999&sort=asc&apikey={etherscan_api_key}"
    
    # Token Transactions (ERC-20) URL
    url = f"https://api.etherscan.io/api?module=account&action=tokentx&address={address}&startblock=0&endblock=99999999&sort=asc&apikey={etherscan_api_key}"
    
    response = requests.get(url)
    return response.json()["result"]

# Since we are interested in UNREALIZED REVENUE GAINS we can sum up 
# differences corresponding to the same token-type, and convert them
# using today's exchange rates post-summation
def organize_transactions_by_token(transactions, address):
    incoming = {}  # Dictionary for incoming token transactions
    outgoing = {}  # Dictionary for outgoing token transactions

    for tx in transactions:
        token_type = tx['tokenName']  # Get the token name
        value = int(tx['value']) / (10 ** int(tx['tokenDecimal']))  # Convert token value using decimals
        
        # Check if the transaction is incoming or outgoing
        if tx['to'].lower() == address.lower():
            # Incoming token transaction
            if token_type not in incoming:
                incoming[token_type] = 0
            incoming[token_type] += value
        elif tx['from'].lower() == address.lower():
            # Outgoing token transaction
            if token_type not in outgoing:
                outgoing[token_type] = 0
            outgoing[token_type] += value
    
    return incoming, outgoing

# Here we simply do the summation which is permissible under the UNREALIZED GAINS definition
def calculate_net_tokens(incoming, outgoing):
    net_totals = {}  # Dictionary to store net token totals
    
    # Iterate through all token types in the incoming dictionary
    for token, incoming_total in incoming.items():
        outgoing_total = outgoing.get(token, 0)  # Get the outgoing total, default to 0 if token not in outgoing
        net_totals[token] = incoming_total - outgoing_total  # Calculate the net total
    
    return net_totals

# Now we put our functions together in a pipeline to find calculate Jared's sandwiching revenue
def main():
    transactions = get_transactions(jared_address)
    incoming, outgoing = organize_transactions_by_token(transactions, jared_address)

    print("Incoming token transactions:")
    for token, total in incoming.items():
        print(f"{token}: {total}")

    print("\nOutgoing token transactions:")
    for token, total in outgoing.items():
        print(f"{token}: {total}")

    # Calculate net tokens
    net_totals = calculate_net_tokens(incoming, outgoing)
    print("\nNet token totals:")
    for token, total in net_totals.items():
        print(f"{token}: {total}")

# Execute the pipeline
main()

