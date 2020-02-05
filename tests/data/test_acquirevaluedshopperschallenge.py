from pprint import pprint

from tests.data.test_DatabaseDataset import TestDataBaseClass


class TestAcquireValuedShoppersChallengeData(TestDataBaseClass.TestData):
    db_name = 'acquirevaluedshopperschallenge'

    def test_preprocessed_datapoint_counts(self):
        true_total_dp = 311541
        self.assertEqual(true_total_dp, sum(len(l) for l in self.loaders.values()))
        self.assertEqual(true_total_dp, sum(len(l.dataset) for l in self.loaders.values()))
        self.assertEqual(true_total_dp, len(self.db_info['train_dp_ids']) + len(self.db_info['test_dp_ids']))

        true_test_dp = 151484
        self.assertEqual(true_test_dp, len(self.loaders['test']))
        self.assertEqual(true_test_dp, len(self.loaders['test'].dataset))
        self.assertEqual(true_test_dp, len(self.db_info['test_dp_ids']))

        true_train_dp = true_total_dp - true_test_dp
        self.assertEqual(true_train_dp, len(self.loaders['train']) + len(self.loaders['val']))
        self.assertEqual(true_train_dp, len(self.loaders['train'].dataset) + len(self.loaders['val'].dataset))
        self.assertEqual(true_train_dp, len(self.db_info['train_dp_ids']))
        self.assertEqual(round(true_train_dp * 0.85), len(self.loaders['train']))

    def test_visually_inspect_this_one_datapoint_in_particular(self):
        dp_id, (edge_list, node_types, edge_types, features, label) = self.loaders['train'].dataset[49147]

        self.assertEqual(dp_id, 15738658)
        history_csv_header = 'id,chain,offer,market,repeattrips,repeater,offerdate'
        history_csv_row = '15738658,17,1197502,4,0,f,2013-04-22'
        pprint({k: (v, features['History'].get(k)) for k, v in
                zip(history_csv_header.split(','), history_csv_row.split(','))})
        offer_csv_header = 'offer,category,quantity,company,offervalue,brand'
        offer_csv_row = '1197502,3203,1,106414464,0.75,13474'
        pprint({k: (v, features['Offer'].get(k)) for k, v in
                zip(offer_csv_header.split(','), offer_csv_row.split(','))})

        self.assertEqual(node_types,
                         [1, 2, 5, 6, 3, 4, 2, 6, 5, 4, 2, 6, 5, 4, 2, 4, 6, 5, 2, 4, 6, 5, 2, 6, 5, 4, 2, 4, 6, 5, 2,
                          4, 5, 6, 2, 6, 5, 4, 2, 6, 4, 5, 2, 4, 5, 2, 6, 4, 5, 2, 6, 5, 2, 6, 5, 4, 2, 6, 5, 2, 6, 5,
                          4, 2, 4, 6, 5, 2, 5, 4, 6, 2, 5, 6, 4, 2, 6, 5, 2, 5, 6, 2, 4, 6, 5, 2, 6, 5, 4, 2, 6, 4, 5,
                          2, 4, 6, 5, 2, 2, 2, 4, 2, 4, 5, 6, 2, 4, 6, 5, 2, 4, 5, 2, 4, 5, 6, 2, 2, 4, 2, 2, 5, 6, 4,
                          2, 6, 5, 4, 2, 6, 4, 2, 5, 6, 0, 5, 6, 4])
        self.assertEqual(edge_list,
                         [(1, 0), (1, 2), (1, 3), (1, 4), (1, 5), (6, 0), (6, 7), (6, 8), (6, 4), (6, 9), (10, 0),
                          (10, 4), (10, 11), (10, 12), (10, 13), (14, 0), (14, 4), (14, 15), (14, 16), (14, 17),
                          (18, 0), (18, 19), (18, 4), (18, 20), (18, 21), (22, 0), (22, 23), (22, 24), (22, 4),
                          (22, 25), (26, 0), (26, 27), (26, 4), (26, 28), (26, 29), (30, 0), (30, 31), (30, 32),
                          (30, 4), (30, 33), (34, 0), (34, 35), (34, 36), (34, 4), (34, 37), (38, 0), (38, 4), (38, 39),
                          (38, 40), (38, 41), (42, 0), (42, 4), (42, 43), (42, 44), (42, 7), (45, 0), (45, 46), (45, 4),
                          (45, 47), (45, 48), (49, 0), (49, 50), (49, 4), (49, 9), (49, 51), (52, 0), (52, 53),
                          (52, 54), (52, 4), (52, 55), (56, 0), (56, 57), (56, 58), (56, 4), (56, 25), (59, 0),
                          (59, 60), (59, 61), (59, 4), (59, 62), (63, 0), (63, 64), (63, 65), (63, 4), (63, 66),
                          (67, 0), (67, 68), (67, 4), (67, 69), (67, 70), (71, 0), (71, 72), (71, 73), (71, 4),
                          (71, 74), (75, 0), (75, 37), (75, 76), (75, 77), (75, 4), (78, 0), (78, 74), (78, 79),
                          (78, 80), (78, 4), (81, 0), (81, 82), (81, 83), (81, 84), (81, 4), (85, 0), (85, 86),
                          (85, 87), (85, 4), (85, 88), (89, 0), (89, 90), (89, 91), (89, 4), (89, 92), (93, 0), (93, 4),
                          (93, 94), (93, 95), (93, 96), (97, 0), (97, 64), (97, 66), (97, 65), (97, 4), (98, 0),
                          (98, 70), (98, 4), (98, 68), (98, 69), (99, 0), (99, 100), (99, 4), (99, 7), (99, 8),
                          (101, 0), (101, 4), (101, 102), (101, 103), (101, 104), (105, 0), (105, 106), (105, 107),
                          (105, 4), (105, 108), (109, 0), (109, 110), (109, 4), (109, 111), (109, 76), (112, 0),
                          (112, 4), (112, 113), (112, 114), (112, 115), (116, 0), (116, 96), (116, 95), (116, 4),
                          (116, 94), (117, 0), (117, 21), (117, 4), (117, 118), (117, 20), (119, 0), (119, 23),
                          (119, 24), (119, 25), (119, 4), (120, 0), (120, 4), (120, 121), (120, 122), (120, 123),
                          (124, 0), (124, 125), (124, 126), (124, 127), (124, 4), (128, 0), (128, 4), (128, 129),
                          (128, 130), (128, 51), (131, 0), (131, 4), (131, 132), (131, 69), (131, 133), (0, 4),
                          (0, 134), (134, 135), (134, 136), (134, 137)]
                         )
        self.assertEqual(edge_types,
                         [6, 10, 9, 7, 8, 6, 9, 10, 7, 8, 6, 7, 9, 10, 8, 6, 7, 8, 9, 10, 6, 8, 7, 9, 10, 6, 9, 10, 7,
                          8, 6, 8, 7, 9, 10, 6, 8, 10, 7, 9, 6, 9, 10, 7, 8, 6, 7, 9, 8, 10, 6, 7, 8, 10, 9, 6, 9, 7, 8,
                          10, 6, 9, 7, 8, 10, 6, 9, 10, 7, 8, 6, 9, 10, 7, 8, 6, 9, 10, 7, 8, 6, 8, 9, 7, 10, 6, 10, 7,
                          8, 9, 6, 10, 9, 7, 8, 6, 8, 9, 10, 7, 6, 8, 10, 9, 7, 6, 8, 9, 10, 7, 6, 9, 10, 7, 8, 6, 9, 8,
                          7, 10, 6, 7, 8, 9, 10, 6, 8, 10, 9, 7, 6, 9, 7, 10, 8, 6, 8, 7, 9, 10, 6, 7, 8, 10, 9, 6, 8,
                          9, 7, 10, 6, 8, 7, 10, 9, 6, 7, 8, 10, 9, 6, 10, 9, 7, 8, 6, 10, 7, 8, 9, 6, 9, 10, 8, 7, 6,
                          7, 10, 9, 8, 6, 9, 10, 8, 7, 6, 7, 9, 8, 10, 6, 7, 10, 8, 9, 5, 4, 3, 2, 1]
                         )
        self.assertEqual(59558 - 59519,
                         len(features['Transaction']['dept']))  # values taken from transactions.csv file
        self.assertEqual(len(features['Category']['category_id']), 31)  # values taken from neo4j
        self.assertEqual(len(features['Chain']['chain_id']), 1)  # values taken from neo4j
        self.assertEqual(len(features['Company']['company_id']), 32)  # values taken from neo4j
        self.assertEqual(len(features['Brand']['brand_id']), 33)  # values taken from neo4j
        self.assertEqual(label, False)
