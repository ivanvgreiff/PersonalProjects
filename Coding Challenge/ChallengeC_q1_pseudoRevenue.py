from web3 import Web3
import requests
import random

"""
This Python script calculates the total UNREALIZED revenue in units of [USD]
It consider's Jared's bot's sandwich trades

Since I do not have access to CoinGecko's massive dictionary mapping the token name to the token ID, 
I simply create fake exchange rates by mapping 1 [token-type] to rand(0, 1) [USD]
Though in reality, most tokens are worth much less, so our calculated revenue is going to look HUGE

If we had CoinGecko's dictionary we could simply use Etherscan's returned token-name, use the dictionary
to map it to the token-ID, and then for each token-ID we query CoinGecko for the current exchange rate

If we were intersted in HISTORICAL revenue, then we would not only query CoinGecko using the token-ID, but 
also the timestamp returned to us by Etherscan. We would then need to calculate differences in costs based
on USD valuations fixed at the time in which the sandwich was executed.

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

# Here I create the fake exchange rates, to use the real ones we only need CoinGecko's massive dictionary, 
# where then we can easily query CoinGecko for the actual (current) exchange rate
def create_random_number_mapping(tokens):
    """
    Create a dictionary mapping token names to a random number between 0 and 1.
    """
    token_random_mapping = {token: random.uniform(0, 1) for token in tokens}
    return token_random_mapping

# Here we apply the pseudo-exchange rates to each token
def calculate_total_usd_revenue(net_totals, token_random_mapping):
    """
    Calculate the total revenue in USD by multiplying net token revenues with fake exchange rates.
    """
    total_revenue_usd = 0
    for token, net_revenue in net_totals.items():
        # Use the fake exchange rate from the dictionary
        fake_exchange_rate = token_random_mapping.get(token, 0)  # Default to 0 if no mapping exists
        revenue_usd = net_revenue * fake_exchange_rate
        total_revenue_usd += revenue_usd
        print(f"{token}: Net Revenue = {net_revenue:.2f}, Fake Exchange Rate = {fake_exchange_rate:.2f}, Revenue USD = {revenue_usd:.2f}")
    return total_revenue_usd

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
        print(f"{token}: {total:.2f}")

    # Create a dictionary mapping token names to random numbers (fake exchange rates)
    token_random_mapping = create_random_number_mapping(net_totals.keys())
    print("\nToken Random Number Mapping:")
    for token, value in token_random_mapping.items():
        print(f"{token}: {value:.2f}")

    # Calculate total revenue in USD
    total_revenue_usd = calculate_total_usd_revenue(net_totals, token_random_mapping)
    print(f"\nTotal Revenue in USD: ${total_revenue_usd:.2f}")

# Execute pipeline
main()
