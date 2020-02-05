from pprint import pprint

from tests.data.test_DatabaseDataset import TestDataBaseClass


class TestKDDCup2014Data(TestDataBaseClass.TestData):
    db_name = 'kddcup2014'

    def test_preprocessed_datapoint_counts(self):
        true_total_dp = 664098
        self.assertEqual(true_total_dp, sum(len(l) for l in self.loaders.values()))
        self.assertEqual(true_total_dp, sum(len(l.dataset) for l in self.loaders.values()))
        self.assertEqual(true_total_dp, len(self.db_info['train_dp_ids']) + len(self.db_info['test_dp_ids']))

        true_test_dp = 44772
        self.assertEqual(true_test_dp, len(self.loaders['test']))
        self.assertEqual(true_test_dp, len(self.loaders['test'].dataset))
        self.assertEqual(true_test_dp, len(self.db_info['test_dp_ids']))

        true_train_dp = true_total_dp - true_test_dp
        self.assertEqual(true_train_dp, len(self.loaders['train']) + len(self.loaders['val']))
        self.assertEqual(true_train_dp, len(self.loaders['train'].dataset) + len(self.loaders['val'].dataset))
        self.assertEqual(true_train_dp, len(self.db_info['train_dp_ids']))
        self.assertEqual(round(true_train_dp * 0.85), len(self.loaders['train']))

    def test_visually_inspect_this_one_datapoint_in_particular(self):
        dp_id, (edge_list, node_types, edge_types, features, label) = self.loaders['train'].dataset[35727]

        self.assertEqual(dp_id, '042f6320073209a847fd59adcb3d1f67')
        project_csv_header = 'projectid,teacher_acctid,schoolid,school_ncesid,school_latitude,school_longitude,school_city,school_state,school_zip,school_metro,school_district,school_county,school_charter,school_magnet,school_year_round,school_nlns,school_kipp,school_charter_ready_promise,teacher_prefix,teacher_teach_for_america,teacher_ny_teaching_fellow,primary_focus_subject,primary_focus_area,secondary_focus_subject,secondary_focus_area,resource_type,poverty_level,grade_level,fulfillment_labor_materials,total_price_excluding_optional_support,total_price_including_optional_support,students_reached,eligible_double_your_impact_match,eligible_almost_home_match,date_posted'
        project_csv_row = '042f6320073209a847fd59adcb3d1f67,9ab72c55783bff5820be2f1b14c0d67b,10a7a9f3f1a9a8660b4c5e70d411f070,063583006679,38.480026,-122.659660,Santa Rosa,CA,95409,urban,Santa Rosa City School Dist,Sonoma,f,f,f,f,f,f,Ms.,f,f,Literature & Writing,Literacy & Language,,,Books,moderate poverty,Grades 9-12,35.00,454.04,534.16,165,f,t,2010-11-21'
        pprint({k: (v, features['Project'].get(k)) for k, v in
                zip(project_csv_header.split(','), project_csv_row.split(','))})

        essay_csv_row = '''
        042f6320073209a847fd59adcb3d1f67,9ab72c55783bff5820be2f1b14c0d67b,Creating A Classroom Library II: G-S,"I don't want my students' only experience with reading to be books assigned that they analyze to death. I want them to see that books can be fun, entertaining, wonderful, and informative--all in one! By adding modern texts to my classroom library, all of my students will...","My students need fun, modern, relevant books to read for fun!","Do you remember the first book you read that you actually enjoyed? It can be life-altering because for the first time, reading is FUN!  ^M\n^M\nI teach grades nine through twelve, both academic and struggling students. I want my students to develop a love of reading so that they can make use of this rich resource. I don't want their only experience with reading to be books assigned that they analyze to death. I want them to see that books can be fun, entertaining, wonderful, and informative--all in one!^M\n^M\nBy adding modern texts to my classroom library, all of my students will have access to reading the latest best fiction as determined by YALSA (Young Adult Library Services Association.) These books include the top ten teen favorites as well as books that have received literary acclaim. All books will be available to any student who wants to borrow and read them. They will also be used in Sustained Silent Reading time when students have time in class to read for fun as well.  ^M\n^M\nBy donating to this whole_project, you will give my students access to the newest books--books that are current, relevant, and fun in their eyes. By providing these books, we may just hook a teen reader for life. "'''
        pprint(essay_csv_row)
        pprint(features['Essay'])

        self.assertEqual(node_types,
                         [0, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                          2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]
                         )
        self.assertEqual(edge_list,
                         [(1, 0), (1, 2), (3, 0), (3, 2), (4, 0), (4, 2), (5, 0), (5, 2), (6, 0), (6, 2), (7, 0),
                          (7, 2), (8, 0), (8, 2), (9, 0), (9, 2), (10, 0), (10, 2), (11, 0), (11, 2), (12, 0), (12, 2),
                          (13, 0), (13, 2), (14, 0), (14, 2), (15, 0), (15, 2), (16, 0), (16, 2), (17, 0), (17, 2),
                          (18, 0), (18, 2), (19, 0), (19, 2), (20, 0), (20, 2), (21, 0), (21, 2), (22, 0), (22, 2),
                          (23, 0), (23, 2), (24, 0), (24, 2), (25, 0), (25, 2), (26, 0), (26, 2), (27, 0), (27, 2),
                          (28, 0), (28, 2), (29, 0), (29, 2), (30, 0), (30, 2), (31, 0), (31, 2), (32, 0), (32, 2),
                          (33, 0), (33, 2), (34, 0), (34, 2), (35, 0), (35, 2), (36, 0), (36, 2), (37, 0), (37, 2),
                          (38, 0), (38, 2), (39, 0), (39, 2), (40, 0), (40, 2), (41, 0), (41, 2), (42, 0), (42, 2),
                          (43, 0), (43, 2), (44, 0), (44, 2), (45, 0), (0, 2)]
                         )
        self.assertEqual(edge_types,
                         [3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3,
                          4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4,
                          3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 2, 1]
                         )
        self.assertEqual(len(features['Essay']['title']), 1)  # values taken from essays.csv
        self.assertEqual(len(features['Resource']['vendor_name']), 3608103 - 3608060)  # values taken from resources.csv
        self.assertEqual(features['ResourceType']['resource_type'], ['Books'])  # values taken from resources.csv
        self.assertEqual(features['Project']['school_id'],
                         ['10a7a9f3f1a9a8660b4c5e70d411f070'])  # values taken from projects.csv
        self.assertEqual(label, False)
