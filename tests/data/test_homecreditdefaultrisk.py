from pprint import pprint

from tests.data.test_DatabaseDataset import TestDataBaseClass


class TestHomeCreditDefaultRiskData(TestDataBaseClass.TestData):
    db_name = 'homecreditdefaultrisk'

    def test_preprocessed_datapoint_counts(self):
        true_total_dp = 356255
        self.assertEqual(true_total_dp, sum(len(l) for l in self.loaders.values()))
        self.assertEqual(true_total_dp, sum(len(l.dataset) for l in self.loaders.values()))
        self.assertEqual(true_total_dp, len(self.db_info['train_dp_ids']) + len(self.db_info['test_dp_ids']))

        true_test_dp = 48744
        self.assertEqual(true_test_dp, len(self.loaders['test']))
        self.assertEqual(true_test_dp, len(self.loaders['test'].dataset))
        self.assertEqual(true_test_dp, len(self.db_info['test_dp_ids']))

        true_train_dp = true_total_dp - true_test_dp
        self.assertEqual(true_train_dp, len(self.loaders['train']) + len(self.loaders['val']))
        self.assertEqual(true_train_dp, len(self.loaders['train'].dataset) + len(self.loaders['val'].dataset))
        self.assertEqual(true_train_dp, len(self.db_info['train_dp_ids']))
        self.assertEqual(round(true_train_dp * 0.85), len(self.loaders['train']))

    def test_visually_inspect_this_one_datapoint_in_particular(self):
        dp_id, (edge_list, node_types, edge_types, features, label) = self.loaders['train'].dataset[235592]

        self.assertEqual(dp_id, 100050)
        application_csv_header = '''SK_ID_CURR,TARGET,NAME_CONTRACT_TYPE,CODE_GENDER,FLAG_OWN_CAR,FLAG_OWN_REALTY,CNT_CHILDREN,AMT_INCOME_TOTAL,AMT_CREDIT,AMT_ANNUITY,AMT_GOODS_PRICE,NAME_TYPE_SUITE,NAME_INCOME_TYPE,NAME_EDUCATION_TYPE,NAME_FAMILY_STATUS,NAME_HOUSING_TYPE,REGION_POPULATION_RELATIVE,DAYS_BIRTH,DAYS_EMPLOYED,DAYS_REGISTRATION,DAYS_ID_PUBLISH,OWN_CAR_AGE,FLAG_MOBIL,FLAG_EMP_PHONE,FLAG_WORK_PHONE,FLAG_CONT_MOBILE,FLAG_PHONE,FLAG_EMAIL,OCCUPATION_TYPE,CNT_FAM_MEMBERS,REGION_RATING_CLIENT,REGION_RATING_CLIENT_W_CITY,WEEKDAY_APPR_PROCESS_START,HOUR_APPR_PROCESS_START,REG_REGION_NOT_LIVE_REGION,REG_REGION_NOT_WORK_REGION,LIVE_REGION_NOT_WORK_REGION,REG_CITY_NOT_LIVE_CITY,REG_CITY_NOT_WORK_CITY,LIVE_CITY_NOT_WORK_CITY,ORGANIZATION_TYPE,EXT_SOURCE_1,EXT_SOURCE_2,EXT_SOURCE_3,APARTMENTS_AVG,BASEMENTAREA_AVG,YEARS_BEGINEXPLUATATION_AVG,YEARS_BUILD_AVG,COMMONAREA_AVG,ELEVATORS_AVG,ENTRANCES_AVG,FLOORSMAX_AVG,FLOORSMIN_AVG,LANDAREA_AVG,LIVINGAPARTMENTS_AVG,LIVINGAREA_AVG,NONLIVINGAPARTMENTS_AVG,NONLIVINGAREA_AVG,APARTMENTS_MODE,BASEMENTAREA_MODE,YEARS_BEGINEXPLUATATION_MODE,YEARS_BUILD_MODE,COMMONAREA_MODE,ELEVATORS_MODE,ENTRANCES_MODE,FLOORSMAX_MODE,FLOORSMIN_MODE,LANDAREA_MODE,LIVINGAPARTMENTS_MODE,LIVINGAREA_MODE,NONLIVINGAPARTMENTS_MODE,NONLIVINGAREA_MODE,APARTMENTS_MEDI,BASEMENTAREA_MEDI,YEARS_BEGINEXPLUATATION_MEDI,YEARS_BUILD_MEDI,COMMONAREA_MEDI,ELEVATORS_MEDI,ENTRANCES_MEDI,FLOORSMAX_MEDI,FLOORSMIN_MEDI,LANDAREA_MEDI,LIVINGAPARTMENTS_MEDI,LIVINGAREA_MEDI,NONLIVINGAPARTMENTS_MEDI,NONLIVINGAREA_MEDI,FONDKAPREMONT_MODE,HOUSETYPE_MODE,TOTALAREA_MODE,WALLSMATERIAL_MODE,EMERGENCYSTATE_MODE,OBS_30_CNT_SOCIAL_CIRCLE,DEF_30_CNT_SOCIAL_CIRCLE,OBS_60_CNT_SOCIAL_CIRCLE,DEF_60_CNT_SOCIAL_CIRCLE,DAYS_LAST_PHONE_CHANGE,FLAG_DOCUMENT_2,FLAG_DOCUMENT_3,FLAG_DOCUMENT_4,FLAG_DOCUMENT_5,FLAG_DOCUMENT_6,FLAG_DOCUMENT_7,FLAG_DOCUMENT_8,FLAG_DOCUMENT_9,FLAG_DOCUMENT_10,FLAG_DOCUMENT_11,FLAG_DOCUMENT_12,FLAG_DOCUMENT_13,FLAG_DOCUMENT_14,FLAG_DOCUMENT_15,FLAG_DOCUMENT_16,FLAG_DOCUMENT_17,FLAG_DOCUMENT_18,FLAG_DOCUMENT_19,FLAG_DOCUMENT_20,FLAG_DOCUMENT_21,AMT_REQ_CREDIT_BUREAU_HOUR,AMT_REQ_CREDIT_BUREAU_DAY,AMT_REQ_CREDIT_BUREAU_WEEK,AMT_REQ_CREDIT_BUREAU_MON,AMT_REQ_CREDIT_BUREAU_QRT,AMT_REQ_CREDIT_BUREAU_YEAR'''
        application_csv_row = '''100050,0,Cash loans,F,N,Y,0,108000.0,746280.0,42970.5,675000.0,Unaccompanied,Pensioner,Higher education,Single / not married,House / apartment,0.010966,-23548,365243,-5745.0,-4576,,1,0,0,1,0,0,,1.0,2,2,WEDNESDAY,9,0,0,0,0,0,0,XNA,,0.7661378050275851,0.6848276586890367,0.2186,0.1232,0.9851,0.7959999999999999,0.0528,0.24,0.2069,0.3333,0.375,0.1154,0.1774,0.2113,0.0039,0.0051,0.2227,0.1279,0.9851,0.804,0.0533,0.2417,0.2069,0.3333,0.375,0.1181,0.1938,0.2202,0.0039,0.0054,0.2207,0.1232,0.9851,0.7987,0.0531,0.24,0.2069,0.3333,0.375,0.1175,0.1804,0.2151,0.0039,0.0052,reg oper spec account,block of flats,0.1903,Panel,No,0.0,0.0,0.0,0.0,-491.0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0,0.0,0.0,0.0,0.0,3.0'''
        pprint({k: (v, features['Application'].get(k)) for k, v in
                zip(application_csv_header.split(','), application_csv_row.split(','))})

        self.assertEqual(node_types,
                         [0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2,
                          2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                          2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                          2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2,
                          2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                          2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                          2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 5, 5, 5, 5, 5, 5, 5, 3, 4, 4,
                          4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
                         )
        self.assertEqual(edge_list,
                         [(1, 0), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (11, 1),
                          (12, 1), (13, 1), (14, 1), (15, 1), (16, 1), (17, 1), (18, 1), (19, 1), (20, 1), (21, 1),
                          (22, 1), (23, 1), (24, 1), (25, 0), (26, 25), (27, 25), (28, 25), (29, 25), (30, 25),
                          (31, 25), (32, 25), (33, 25), (34, 25), (35, 25), (36, 25), (37, 25), (38, 25), (39, 25),
                          (40, 25), (41, 25), (42, 25), (43, 25), (44, 25), (45, 25), (46, 25), (47, 25), (48, 25),
                          (49, 25), (50, 25), (51, 25), (52, 25), (53, 25), (54, 25), (55, 25), (56, 25), (57, 25),
                          (58, 25), (59, 25), (60, 25), (61, 25), (62, 25), (63, 25), (64, 25), (65, 25), (66, 25),
                          (67, 25), (68, 25), (69, 25), (70, 25), (71, 25), (72, 25), (73, 25), (74, 25), (75, 25),
                          (76, 25), (77, 25), (78, 25), (79, 25), (80, 25), (81, 25), (82, 25), (83, 25), (84, 25),
                          (85, 25), (86, 25), (87, 25), (88, 25), (89, 25), (90, 25), (91, 25), (92, 25), (93, 25),
                          (94, 25), (95, 25), (96, 25), (97, 25), (98, 25), (99, 25), (100, 25), (101, 25), (102, 25),
                          (103, 25), (104, 25), (105, 25), (106, 25), (107, 25), (108, 25), (109, 25), (110, 25),
                          (111, 25), (112, 25), (113, 25), (114, 25), (115, 0), (116, 115), (117, 115), (118, 115),
                          (119, 115), (120, 115), (121, 115), (122, 115), (123, 115), (124, 115), (125, 115),
                          (126, 115), (127, 115), (128, 115), (129, 115), (130, 115), (131, 115), (132, 115),
                          (133, 115), (134, 115), (135, 115), (136, 115), (137, 115), (138, 115), (139, 115),
                          (140, 115), (141, 115), (142, 115), (143, 115), (144, 115), (145, 115), (146, 115),
                          (147, 115), (148, 115), (149, 115), (150, 115), (151, 115), (152, 115), (153, 115),
                          (154, 115), (155, 115), (156, 115), (157, 115), (158, 115), (159, 115), (160, 115),
                          (161, 115), (162, 115), (163, 115), (164, 115), (165, 115), (166, 115), (167, 115),
                          (168, 115), (169, 115), (170, 115), (171, 115), (172, 115), (173, 115), (174, 115),
                          (175, 115), (176, 115), (177, 115), (178, 115), (179, 115), (180, 115), (181, 115),
                          (182, 115), (183, 115), (184, 115), (185, 115), (186, 115), (187, 115), (188, 115),
                          (189, 115), (190, 115), (191, 115), (192, 115), (193, 115), (194, 115), (195, 115),
                          (196, 115), (197, 115), (198, 115), (199, 115), (200, 115), (201, 115), (202, 115),
                          (203, 115), (204, 115), (205, 0), (206, 0), (207, 206), (208, 206), (209, 206), (210, 206),
                          (211, 206), (212, 206), (213, 206), (214, 0), (215, 214), (216, 214), (217, 214), (218, 214),
                          (219, 214), (220, 214), (221, 214), (222, 214), (223, 214), (224, 214), (225, 214),
                          (226, 214), (227, 214), (228, 214), (229, 214), (230, 214), (231, 214), (232, 214),
                          (233, 214), (234, 214), (235, 214), (236, 214), (237, 214), (238, 214), (239, 214),
                          (240, 214), (241, 214), (242, 214), (243, 214), (244, 214), (245, 214), (246, 214), (215, 0),
                          (216, 0), (217, 0), (218, 0), (219, 0), (220, 0), (221, 0), (223, 0), (222, 0), (224, 0),
                          (225, 0), (226, 0), (227, 0), (228, 0), (229, 0), (230, 0), (207, 0), (208, 0), (209, 0),
                          (210, 0), (211, 0), (212, 0), (213, 0), (231, 0), (232, 0), (233, 0), (234, 0), (235, 0),
                          (236, 0), (237, 0), (238, 0), (239, 0), (240, 0), (241, 0), (242, 0), (243, 0), (244, 0),
                          (245, 0), (246, 0)]
                         )
        self.assertEqual(edge_types,
                         [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2,
                          2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                          2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                          2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                          2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                          2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                          2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 7, 7, 7, 7, 7, 7, 7, 3, 5, 5, 5,
                          5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 4, 4,
                          4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
                          8, 8, 8, 8, 8, 8]
                         )
        self.assertEqual(len(features['Bureau']['CREDIT_TYPE']), 3)  # values taken from bureau.csv
        self.assertEqual(len(features['CashBalance']['CNT_INSTALMENT']), 16)  # values taken from POS_CASH_balance.csv
        self.assertEqual(len(features['InstallmentPayment']['AMT_INSTALMENT']),
                         16)  # values taken from installments_payments.csv
        self.assertEqual(len(features['CreditBalance']['AMT_BALANCE']), 7)  # values taken from credit_card_balance.csv
        self.assertEqual(len(features['PreviousApplication']['AMT_ANNUITY']),
                         3)  # values taken from previous_application.csv
        self.assertEqual(len(features['BureauBalance']['MONTHS_BALANCE']), 201)  # values taken from neo4j
        self.assertEqual(label, False)
