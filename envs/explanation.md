Multi-Agent Beer Game Supply Chain Environment (beer_game_v0)
1. System Topology & Objective

The environment simulates a linear, four-stage supply chain facing external customer demand.

    The Agents (Downstream to Upstream): Retailer -> Wholesaler -> Distributor -> Manufacturer.

    The Objective: The environment is fully cooperative. The goal of the multi-agent system is to minimize the Total System Cost across all four nodes over a 50-week horizon.

2. Initialization (The Sterman Steady-State)

At Step 0, the environment initializes in a classic "Sterman Steady-State" to prevent immediate chaotic shocks:

    Initial Inventory: 12 units per agent.

    Initial Backlog: 0 units per agent.

    Pipeline Transit: There are exactly 4 units of beer arriving in 1 week, and 4 units of beer arriving in 2 weeks for every agent. The same is true for the information/order pipelines.

3. The Observation Space (State)

Each agent is partially observable. They cannot see the global supply chain. At the start of each week, an agent receives a 6-dimensional continuous vector containing:

    Current Local Inventory

    Current Local Backlog

    Incoming Shipments arriving in exactly 1 week

    Incoming Shipments arriving in exactly 2 weeks

    Incoming Shipments arriving in exactly 3 weeks

    Incoming Shipments arriving in exactly 4 weeks

4. The Action Space

Each agent outputs a single continuous value bounded between 0.0 and 1.0.

    This value is multiplied by the max_order threshold (Default: 100) and rounded to the nearest integer to determine the physical cases of beer ordered that week.

    Example: An action of 0.125 results in an order of 13 units.

5. The Step Sequence (Rules of Play)

Every single step (week) executes in the following strict chronological order:

    Place Orders (Information Flow Upstream): Agents send their orders to their upstream supplier. These orders are placed in a pipeline with a strict 2-week information delay.

    Receive Shipments (Material Flow Downstream): Agents add any beer arriving this week from their shipment pipeline to their current inventory.

    Determine Demand: The Retailer looks at the external customer market. All other agents look at the orders arriving from their downstream partner's pipeline.

    Fulfill Demand: Agents attempt to fulfill the demand + any existing backlog. If Inventory is greater than or equal to Demand, it is fulfilled. If Inventory is less than Demand, inventory drops to 0 and the remainder is added to the Backlog.

    Ship Beer (Material Flow Downstream): Fulfilled orders are shipped to the downstream agent. These shipments face a strict 2-week material delay (unless the jittery_lead_time stress test is active, which randomizes delays between 1-10 weeks).

    Calculate Costs & Rewards: * Holding Cost: 0.50 per unit of inventory.

        Backorder Cost: 1.00 per unit of backlog.

        Reward: All agents receive a shared reward equal to the negative sum of all agents' costs.

6. The External Market Scenarios

The Retailer faces one of four distinct demand distributions:

    step (Canonical Benchmark): Steady demand of 4 units, permanently jumping to 8 units at week 5.

    poisson (Training Domain): Stochastic demand drawn from a Poisson distribution with a mean of 8.

    black_swan (OOD Shock): Stochastic demand of 8, jumping to a massive mean of 20 at week 25.

    extreme_chaos (Pandemic Shock): Multiphase collapse. Base 8 (weeks 0-9) -> Panic Buy of 30 (weeks 10-19) -> Market Freeze of 0 (weeks 20-29) -> Random Whiplash of 5 to 25 (weeks 30-50).