from data.utils import build_db_info

db_info = {'task': {'type': 'classification',
                    'n_classes': 2,
                    'n_train': 160057,
                    'n_test': 151484,
                    'train_class_counts': [116619, 43438]
                    },
           'node_type_to_int': {
               'Offer': 0,
               'History': 1,
               'Transaction': 2,
               'Chain': 3,
               'Category': 4,
               'Brand': 5,
               'Company': 6
           },
           'edge_type_to_int': {
               'SELF': 0,
               'OFFER_TO_CATEGORY': 1,
               'OFFER_TO_COMPANY': 2,
               'OFFER_TO_BRAND': 3,
               'HISTORY_TO_OFFER': 4,
               'HISTORY_TO_CHAIN': 5,
               'TRANSACTION_TO_HISTORY': 6,
               'TRANSACTION_TO_CHAIN': 7,
               'TRANSACTION_TO_CATEGORY': 8,
               'TRANSACTION_TO_COMPANY': 9,
               'TRANSACTION_TO_BRAND': 10
           },
           'node_types_and_features': {'Offer': {'offer_id': {'type': 'CATEGORICAL'},
                                                 'quantity': {'type': 'SCALAR'},
                                                 'offervalue': {'type': 'SCALAR'},
                                                 },
                                       'History': {'market': {'type': 'CATEGORICAL'},
                                                   'repeater': {'type': 'CATEGORICAL'},
                                                   'offerdate': {'type': 'DATETIME'},
                                                   },
                                       'Transaction': {'dept': {'type': 'CATEGORICAL'},
                                                       'date': {'type': 'DATETIME'},
                                                       'productsize': {'type': 'SCALAR'},
                                                       'productmeasure': {'type': 'CATEGORICAL'},
                                                       'purchasequantity': {'type': 'SCALAR'},
                                                       'purchaseamount': {'type': 'SCALAR'},
                                                       },
                                       'Chain': {'chain_id': {'type': 'CATEGORICAL'},
                                                 },
                                       'Category': {'category_id': {'type': 'CATEGORICAL'},
                                                    },
                                       'Brand': {'brand_id': {'type': 'CATEGORICAL'},
                                                 },
                                       'Company': {'company_id': {'type': 'CATEGORICAL'},
                                                   },
                                       },
           'label_feature': 'History.repeater'
           }

if __name__ == '__main__':
    db_name = 'acquirevaluedshopperschallenge'
    test_dp_query = 'MATCH (h:History) WHERE NOT EXISTS(h.repeater) RETURN h.history_id ORDER BY h.history_id '
    train_dp_query = 'MATCH (h:History) WHERE EXISTS(h.repeater) RETURN h.history_id ORDER BY h.history_id '
    build_db_info(db_name, db_info, test_dp_query, train_dp_query)
