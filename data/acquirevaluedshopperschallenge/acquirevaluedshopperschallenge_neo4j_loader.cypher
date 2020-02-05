CREATE CONSTRAINT ON (o:Offer) ASSERT o.offer_id IS UNIQUE;

CREATE CONSTRAINT ON (h:History) ASSERT h.history_id IS UNIQUE;

CREATE CONSTRAINT ON (t:Transaction) ASSERT t.transaction_id IS UNIQUE;

CREATE CONSTRAINT ON (ch:Chain) ASSERT ch.chain_id IS UNIQUE;

CREATE CONSTRAINT ON (cat:Category) ASSERT cat.category_id IS UNIQUE;

CREATE CONSTRAINT ON (b:Brand) ASSERT b.brand_id IS UNIQUE;

CREATE CONSTRAINT ON (co:Company) ASSERT co.company_id IS UNIQUE;

CALL db.awaitIndexes(600);

// Create Offers
USING PERIODIC COMMIT
LOAD CSV WITH HEADERS FROM 'file:///data/offers.csv' AS row
CREATE (o:Offer {offer_id:   toInteger(row.offer),
                 quantity:   toFloat(row.quantity),
                 offervalue: toFloat(row.offervalue)})
WITH o, row
MERGE (cat:Category {category_id: toInteger(row.category)})
CREATE (o)-[:OFFER_TO_CATEGORY]->(cat)
MERGE (co:Company {company_id: toInteger(row.company)})
CREATE (o)-[:OFFER_TO_COMPANY]->(co)
MERGE (b:Brand {brand_id: toInteger(row.brand)})
CREATE (o)-[:OFFER_TO_BRAND]->(b);

CALL db.awaitIndexes(600);

// Create Histories
USING PERIODIC COMMIT
LOAD CSV WITH HEADERS FROM 'file:///data/trainHistory.csv' AS row
CREATE (h:History {history_id: toInteger(row.id),
                   market:     toInteger(row.market),
//                 repeattrips: toFloat(row.repeattrips), // only exists for training points
                   repeater:   toBoolean(replace(replace(row.repeater, 't', 'TRUE'), 'f', 'FALSE')),
                   offerdate:  date(row.offerdate)})
WITH h, row
MERGE (o:Offer {offer_id: toInteger(row.offer)})
CREATE (h)-[:HISTORY_TO_OFFER]->(o)
MERGE (ch:Chain {chain_id: toInteger(row.chain)})
CREATE (h)-[:HISTORY_TO_CHAIN]->(ch);

CALL db.awaitIndexes(600);

USING PERIODIC COMMIT
LOAD CSV WITH HEADERS FROM 'file:///data/testHistory.csv' AS row
CREATE (h:History {history_id: toInteger(row.id),
                   market:     toInteger(row.market),
                   offerdate:  date(row.offerdate)})
WITH h, row
MERGE (o:Offer {offer_id: toInteger(row.offer)})
CREATE (h)-[:HISTORY_TO_OFFER]->(o)
MERGE (ch:Chain {chain_id: toInteger(row.chain)})
CREATE (h)-[:HISTORY_TO_CHAIN]->(ch);

CALL db.awaitIndexes(600);

// Create Transactions
USING PERIODIC COMMIT 10000
LOAD CSV WITH HEADERS FROM 'file:///data/temp.csv' AS row
CREATE (t:Transaction {transaction_id:   toInteger(row.id),
                       dept:             toInteger(row.dept),
                       date:             date(row.date),
                       productsize:      toFloat(row.productsize),
                       productmeasure:   toString(row.productmeasure),
                       purchasequantity: toFloat(row.purchasequantity),
                       purchaseamount:   toFloat(row.purchaseamount)})
WITH t, row
MERGE (h:History {history_id: toInteger(row.history)})
CREATE (t)-[:TRANSACTION_TO_HISTORY]->(h)
MERGE (ch:Chain {chain_id: toInteger(row.chain)})
CREATE (t)-[:TRANSACTION_TO_CHAIN]->(ch)
MERGE (cat:Category {category_id: toInteger(row.category)})
CREATE (t)-[:TRANSACTION_TO_CATEGORY]->(cat)
MERGE (co:Company {company_id: toInteger(row.company)})
CREATE (t)-[:TRANSACTION_TO_COMPANY]->(co)
MERGE (b:Brand {brand_id: toInteger(row.brand)})
CREATE (t)-[:TRANSACTION_TO_BRAND]->(b);

CALL db.awaitIndexes(600);
