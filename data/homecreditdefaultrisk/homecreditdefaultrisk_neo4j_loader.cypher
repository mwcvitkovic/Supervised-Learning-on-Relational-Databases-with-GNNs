CREATE CONSTRAINT ON (app:Application) ASSERT app.SK_ID_CURR IS UNIQUE;

CREATE CONSTRAINT ON (bur:Bureau) ASSERT bur.SK_ID_BUREAU IS UNIQUE;

CREATE CONSTRAINT ON (papp:PreviousApplication) ASSERT papp.SK_ID_PREV IS UNIQUE;

// Create Applications (Test)
USING PERIODIC COMMIT
LOAD CSV WITH HEADERS FROM 'file:///data/application_test.csv' AS row
CREATE (app:Application {SK_ID_CURR:                   toInteger(row.SK_ID_CURR),
                         TARGET:                       null,
                         NAME_CONTRACT_TYPE:           row.NAME_CONTRACT_TYPE,
                         CODE_GENDER:                  row.CODE_GENDER,
                         FLAG_OWN_CAR:                 toBoolean(replace(replace(row.
  FLAG_OWN_CAR, 'Y', 'TRUE'), 'N', 'FALSE')),
                         FLAG_OWN_REALTY:              toBoolean(replace(replace(row.
  FLAG_OWN_REALTY, 'Y', 'TRUE'), 'N', 'FALSE')),
                         CNT_CHILDREN:                 toFloat(row.CNT_CHILDREN),
                         AMT_INCOME_TOTAL:             toFloat(row.AMT_INCOME_TOTAL),
                         AMT_CREDIT:                   toFloat(row.AMT_CREDIT),
                         AMT_ANNUITY:                  toFloat(row.AMT_ANNUITY),
                         AMT_GOODS_PRICE:              toFloat(row.AMT_GOODS_PRICE),
                         NAME_TYPE_SUITE:              row.NAME_TYPE_SUITE,
                         NAME_INCOME_TYPE:             row.NAME_INCOME_TYPE,
                         NAME_EDUCATION_TYPE:          row.NAME_EDUCATION_TYPE,
                         NAME_FAMILY_STATUS:           row.NAME_FAMILY_STATUS,
                         NAME_HOUSING_TYPE:            row.NAME_HOUSING_TYPE,
                         REGION_POPULATION_RELATIVE:   toFloat(row.REGION_POPULATION_RELATIVE),
                         DAYS_BIRTH:                   -1 * toFloat(row.DAYS_BIRTH), // encoded wrong in dataset
                         DAYS_EMPLOYED:                toFloat(replace(row.
  DAYS_EMPLOYED, '365243', 'null')), // 365243 means null
                         DAYS_REGISTRATION:            toFloat(row.DAYS_REGISTRATION),
                         DAYS_ID_PUBLISH:              toFloat(row.DAYS_ID_PUBLISH),
                         OWN_CAR_AGE:                  toFloat(row.OWN_CAR_AGE),
                         FLAG_MOBIL:                   toBoolean(replace(replace(row.
  FLAG_MOBIL, '1', 'TRUE'), '0', 'FALSE')),
                         FLAG_EMP_PHONE:               toBoolean(replace(replace(row.
  FLAG_EMP_PHONE, '1', 'TRUE'), '0', 'FALSE')),
                         FLAG_WORK_PHONE:              toBoolean(replace(replace(row.
  FLAG_WORK_PHONE, '1', 'TRUE'), '0', 'FALSE')),
                         FLAG_CONT_MOBILE:             toBoolean(replace(replace(row.
  FLAG_CONT_MOBILE, '1', 'TRUE'), '0', 'FALSE')),
                         FLAG_PHONE:                   toBoolean(replace(replace(row.
  FLAG_PHONE, '1', 'TRUE'), '0', 'FALSE')),
                         FLAG_EMAIL:                   toBoolean(replace(replace(row.
  FLAG_EMAIL, '1', 'TRUE'), '0', 'FALSE')),
                         OCCUPATION_TYPE:              row.OCCUPATION_TYPE,
                         CNT_FAM_MEMBERS:              toFloat(row.CNT_FAM_MEMBERS),
                         REGION_RATING_CLIENT:         toInteger(row.REGION_RATING_CLIENT),
                         REGION_RATING_CLIENT_W_CITY:  toInteger(row.REGION_RATING_CLIENT_W_CITY),
                         WEEKDAY_APPR_PROCESS_START:   row.WEEKDAY_APPR_PROCESS_START,
                         HOUR_APPR_PROCESS_START:      toInteger(row.HOUR_APPR_PROCESS_START),
                         REG_REGION_NOT_LIVE_REGION:   toBoolean(replace(replace(row.
  REG_REGION_NOT_LIVE_REGION, '1', 'TRUE'), '0', 'FALSE')),
                         REG_REGION_NOT_WORK_REGION:   toBoolean(replace(replace(row.
  REG_REGION_NOT_WORK_REGION, '1', 'TRUE'), '0', 'FALSE')),
                         LIVE_REGION_NOT_WORK_REGION:  toBoolean(replace(replace(row.
  LIVE_REGION_NOT_WORK_REGION, '1', 'TRUE'), '0', 'FALSE')),
                         REG_CITY_NOT_LIVE_CITY:       toBoolean(replace(replace(row.
  REG_CITY_NOT_LIVE_CITY, '1', 'TRUE'), '0', 'FALSE')),
                         REG_CITY_NOT_WORK_CITY:       toBoolean(replace(replace(row.
  REG_CITY_NOT_WORK_CITY, '1', 'TRUE'), '0', 'FALSE')),
                         LIVE_CITY_NOT_WORK_CITY:      toBoolean(replace(replace(row.
  LIVE_CITY_NOT_WORK_CITY, '1', 'TRUE'), '0', 'FALSE')),
                         ORGANIZATION_TYPE:            row.ORGANIZATION_TYPE,
                         EXT_SOURCE_1:                 toFloat(row.EXT_SOURCE_1),
                         EXT_SOURCE_2:                 toFloat(row.EXT_SOURCE_2),
                         EXT_SOURCE_3:                 toFloat(row.EXT_SOURCE_3),
                         APARTMENTS_AVG:               toFloat(row.APARTMENTS_AVG),
                         BASEMENTAREA_AVG:             toFloat(row.BASEMENTAREA_AVG),
                         YEARS_BEGINEXPLUATATION_AVG:  toFloat(row.YEARS_BEGINEXPLUATATION_AVG),
                         YEARS_BUILD_AVG:              toFloat(row.YEARS_BUILD_AVG),
                         COMMONAREA_AVG:               toFloat(row.COMMONAREA_AVG),
                         ELEVATORS_AVG:                toFloat(row.ELEVATORS_AVG),
                         ENTRANCES_AVG:                toFloat(row.ENTRANCES_AVG),
                         FLOORSMAX_AVG:                toFloat(row.FLOORSMAX_AVG),
                         FLOORSMIN_AVG:                toFloat(row.FLOORSMIN_AVG),
                         LANDAREA_AVG:                 toFloat(row.LANDAREA_AVG),
                         LIVINGAPARTMENTS_AVG:         toFloat(row.LIVINGAPARTMENTS_AVG),
                         LIVINGAREA_AVG:               toFloat(row.LIVINGAREA_AVG),
                         NONLIVINGAPARTMENTS_AVG:      toFloat(row.NONLIVINGAPARTMENTS_AVG),
                         NONLIVINGAREA_AVG:            toFloat(row.NONLIVINGAREA_AVG),
                         APARTMENTS_MODE:              toFloat(row.APARTMENTS_MODE),
                         BASEMENTAREA_MODE:            toFloat(row.BASEMENTAREA_MODE),
                         YEARS_BEGINEXPLUATATION_MODE: toFloat(row.YEARS_BEGINEXPLUATATION_MODE),
                         YEARS_BUILD_MODE:             toFloat(row.YEARS_BUILD_MODE),
                         COMMONAREA_MODE:              toFloat(row.COMMONAREA_MODE),
                         ELEVATORS_MODE:               toFloat(row.ELEVATORS_MODE),
                         ENTRANCES_MODE:               toFloat(row.ENTRANCES_MODE),
                         FLOORSMAX_MODE:               toFloat(row.FLOORSMAX_MODE),
                         FLOORSMIN_MODE:               toFloat(row.FLOORSMIN_MODE),
                         LANDAREA_MODE:                toFloat(row.LANDAREA_MODE),
                         LIVINGAPARTMENTS_MODE:        toFloat(row.LIVINGAPARTMENTS_MODE),
                         LIVINGAREA_MODE:              toFloat(row.LIVINGAREA_MODE),
                         NONLIVINGAPARTMENTS_MODE:     toFloat(row.NONLIVINGAPARTMENTS_MODE),
                         NONLIVINGAREA_MODE:           toFloat(row.NONLIVINGAREA_MODE),
                         APARTMENTS_MEDI:              toFloat(row.APARTMENTS_MEDI),
                         BASEMENTAREA_MEDI:            toFloat(row.BASEMENTAREA_MEDI),
                         YEARS_BEGINEXPLUATATION_MEDI: toFloat(row.YEARS_BEGINEXPLUATATION_MEDI),
                         YEARS_BUILD_MEDI:             toFloat(row.YEARS_BUILD_MEDI),
                         COMMONAREA_MEDI:              toFloat(row.COMMONAREA_MEDI),
                         ELEVATORS_MEDI:               toFloat(row.ELEVATORS_MEDI),
                         ENTRANCES_MEDI:               toFloat(row.ENTRANCES_MEDI),
                         FLOORSMAX_MEDI:               toFloat(row.FLOORSMAX_MEDI),
                         FLOORSMIN_MEDI:               toFloat(row.FLOORSMIN_MEDI),
                         LANDAREA_MEDI:                toFloat(row.LANDAREA_MEDI),
                         LIVINGAPARTMENTS_MEDI:        toFloat(row.LIVINGAPARTMENTS_MEDI),
                         LIVINGAREA_MEDI:              toFloat(row.LIVINGAREA_MEDI),
                         NONLIVINGAPARTMENTS_MEDI:     toFloat(row.NONLIVINGAPARTMENTS_MEDI),
                         NONLIVINGAREA_MEDI:           toFloat(row.NONLIVINGAREA_MEDI),
                         FONDKAPREMONT_MODE:           row.FONDKAPREMONT_MODE,
                         HOUSETYPE_MODE:               row.HOUSETYPE_MODE,
                         TOTALAREA_MODE:               toFloat(row.TOTALAREA_MODE),
                         WALLSMATERIAL_MODE:           row.WALLSMATERIAL_MODE,
                         EMERGENCYSTATE_MODE:          row.EMERGENCYSTATE_MODE,
                         OBS_30_CNT_SOCIAL_CIRCLE:     toFloat(row.OBS_30_CNT_SOCIAL_CIRCLE),
                         DEF_30_CNT_SOCIAL_CIRCLE:     toFloat(row.DEF_30_CNT_SOCIAL_CIRCLE),
                         OBS_60_CNT_SOCIAL_CIRCLE:     toFloat(row.OBS_60_CNT_SOCIAL_CIRCLE),
                         DEF_60_CNT_SOCIAL_CIRCLE:     toFloat(row.DEF_60_CNT_SOCIAL_CIRCLE),
                         DAYS_LAST_PHONE_CHANGE:       toFloat(row.DAYS_LAST_PHONE_CHANGE),
                         FLAG_DOCUMENT_2:              toBoolean(replace(replace(row.
  FLAG_DOCUMENT_2, '1', 'TRUE'), '0', 'FALSE')),
                         FLAG_DOCUMENT_3:              toBoolean(replace(replace(row.
  FLAG_DOCUMENT_3, '1', 'TRUE'), '0', 'FALSE')),
                         FLAG_DOCUMENT_4:              toBoolean(replace(replace(row.
  FLAG_DOCUMENT_4, '1', 'TRUE'), '0', 'FALSE')),
                         FLAG_DOCUMENT_5:              toBoolean(replace(replace(row.
  FLAG_DOCUMENT_5, '1', 'TRUE'), '0', 'FALSE')),
                         FLAG_DOCUMENT_6:              toBoolean(replace(replace(row.
  FLAG_DOCUMENT_6, '1', 'TRUE'), '0', 'FALSE')),
                         FLAG_DOCUMENT_7:              toBoolean(replace(replace(row.
  FLAG_DOCUMENT_7, '1', 'TRUE'), '0', 'FALSE')),
                         FLAG_DOCUMENT_8:              toBoolean(replace(replace(row.
  FLAG_DOCUMENT_8, '1', 'TRUE'), '0', 'FALSE')),
                         FLAG_DOCUMENT_9:              toBoolean(replace(replace(row.
  FLAG_DOCUMENT_9, '1', 'TRUE'), '0', 'FALSE')),
                         FLAG_DOCUMENT_10:             toBoolean(replace(replace(row.
  FLAG_DOCUMENT_10, '1', 'TRUE'), '0', 'FALSE')),
                         FLAG_DOCUMENT_11:             toBoolean(replace(replace(row.
  FLAG_DOCUMENT_11, '1', 'TRUE'), '0', 'FALSE')),
                         FLAG_DOCUMENT_12:             toBoolean(replace(replace(row.
  FLAG_DOCUMENT_12, '1', 'TRUE'), '0', 'FALSE')),
                         FLAG_DOCUMENT_13:             toBoolean(replace(replace(row.
  FLAG_DOCUMENT_13, '1', 'TRUE'), '0', 'FALSE')),
                         FLAG_DOCUMENT_14:             toBoolean(replace(replace(row.
  FLAG_DOCUMENT_14, '1', 'TRUE'), '0', 'FALSE')),
                         FLAG_DOCUMENT_15:             toBoolean(replace(replace(row.
  FLAG_DOCUMENT_15, '1', 'TRUE'), '0', 'FALSE')),
                         FLAG_DOCUMENT_16:             toBoolean(replace(replace(row.
  FLAG_DOCUMENT_16, '1', 'TRUE'), '0', 'FALSE')),
                         FLAG_DOCUMENT_17:             toBoolean(replace(replace(row.
  FLAG_DOCUMENT_17, '1', 'TRUE'), '0', 'FALSE')),
                         FLAG_DOCUMENT_18:             toBoolean(replace(replace(row.
  FLAG_DOCUMENT_18, '1', 'TRUE'), '0', 'FALSE')),
                         FLAG_DOCUMENT_19:             toBoolean(replace(replace(row.
  FLAG_DOCUMENT_19, '1', 'TRUE'), '0', 'FALSE')),
                         FLAG_DOCUMENT_20:             toBoolean(replace(replace(row.
  FLAG_DOCUMENT_20, '1', 'TRUE'), '0', 'FALSE')),
                         FLAG_DOCUMENT_21:             toBoolean(replace(replace(row.
  FLAG_DOCUMENT_21, '1', 'TRUE'), '0', 'FALSE')),
                         AMT_REQ_CREDIT_BUREAU_HOUR:   toFloat(row.AMT_REQ_CREDIT_BUREAU_HOUR),
                         AMT_REQ_CREDIT_BUREAU_DAY:    toFloat(row.AMT_REQ_CREDIT_BUREAU_DAY),
                         AMT_REQ_CREDIT_BUREAU_WEEK:   toFloat(row.AMT_REQ_CREDIT_BUREAU_WEEK),
                         AMT_REQ_CREDIT_BUREAU_MON:    toFloat(row.AMT_REQ_CREDIT_BUREAU_MON),
                         AMT_REQ_CREDIT_BUREAU_QRT:    toFloat(row.AMT_REQ_CREDIT_BUREAU_QRT),
                         AMT_REQ_CREDIT_BUREAU_YEAR:   toFloat(row.AMT_REQ_CREDIT_BUREAU_YEAR)
});


// Create Applications (Train)
USING PERIODIC COMMIT
LOAD CSV WITH HEADERS FROM 'file:///data/application_train.csv' AS row
CREATE (app:Application {SK_ID_CURR:                   toInteger(row.SK_ID_CURR),
                         TARGET:                       toBoolean(replace(replace(row.
  TARGET, '1', 'TRUE'), '0', 'FALSE')),
                         NAME_CONTRACT_TYPE:           row.NAME_CONTRACT_TYPE,
                         CODE_GENDER:                  row.CODE_GENDER,
                         FLAG_OWN_CAR:                 toBoolean(replace(replace(row.
  FLAG_OWN_CAR, 'Y', 'TRUE'), 'N', 'FALSE')),
                         FLAG_OWN_REALTY:              toBoolean(replace(replace(row.
  FLAG_OWN_REALTY, 'Y', 'TRUE'), 'N', 'FALSE')),
                         CNT_CHILDREN:                 toFloat(row.CNT_CHILDREN),
                         AMT_INCOME_TOTAL:             toFloat(row.AMT_INCOME_TOTAL),
                         AMT_CREDIT:                   toFloat(row.AMT_CREDIT),
                         AMT_ANNUITY:                  toFloat(row.AMT_ANNUITY),
                         AMT_GOODS_PRICE:              toFloat(row.AMT_GOODS_PRICE),
                         NAME_TYPE_SUITE:              row.NAME_TYPE_SUITE,
                         NAME_INCOME_TYPE:             row.NAME_INCOME_TYPE,
                         NAME_EDUCATION_TYPE:          row.NAME_EDUCATION_TYPE,
                         NAME_FAMILY_STATUS:           row.NAME_FAMILY_STATUS,
                         NAME_HOUSING_TYPE:            row.NAME_HOUSING_TYPE,
                         REGION_POPULATION_RELATIVE:   toFloat(row.REGION_POPULATION_RELATIVE),
                         DAYS_BIRTH:                   -1 * toFloat(row.DAYS_BIRTH), // encoded wrong in dataset
                         DAYS_EMPLOYED:                toFloat(replace(row.
  DAYS_EMPLOYED, '365243', 'null')), // 365243 means null
                         DAYS_REGISTRATION:            toFloat(row.DAYS_REGISTRATION),
                         DAYS_ID_PUBLISH:              toFloat(row.DAYS_ID_PUBLISH),
                         OWN_CAR_AGE:                  toFloat(row.OWN_CAR_AGE),
                         FLAG_MOBIL:                   toBoolean(replace(replace(row.
  FLAG_MOBIL, '1', 'TRUE'), '0', 'FALSE')),
                         FLAG_EMP_PHONE:               toBoolean(replace(replace(row.
  FLAG_EMP_PHONE, '1', 'TRUE'), '0', 'FALSE')),
                         FLAG_WORK_PHONE:              toBoolean(replace(replace(row.
  FLAG_WORK_PHONE, '1', 'TRUE'), '0', 'FALSE')),
                         FLAG_CONT_MOBILE:             toBoolean(replace(replace(row.
  FLAG_CONT_MOBILE, '1', 'TRUE'), '0', 'FALSE')),
                         FLAG_PHONE:                   toBoolean(replace(replace(row.
  FLAG_PHONE, '1', 'TRUE'), '0', 'FALSE')),
                         FLAG_EMAIL:                   toBoolean(replace(replace(row.
  FLAG_EMAIL, '1', 'TRUE'), '0', 'FALSE')),
                         OCCUPATION_TYPE:              row.OCCUPATION_TYPE,
                         CNT_FAM_MEMBERS:              toFloat(row.CNT_FAM_MEMBERS),
                         REGION_RATING_CLIENT:         toInteger(row.REGION_RATING_CLIENT),
                         REGION_RATING_CLIENT_W_CITY:  toInteger(row.REGION_RATING_CLIENT_W_CITY),
                         WEEKDAY_APPR_PROCESS_START:   row.WEEKDAY_APPR_PROCESS_START,
                         HOUR_APPR_PROCESS_START:      toInteger(row.HOUR_APPR_PROCESS_START),
                         REG_REGION_NOT_LIVE_REGION:   toBoolean(replace(replace(row.
  REG_REGION_NOT_LIVE_REGION, '1', 'TRUE'), '0', 'FALSE')),
                         REG_REGION_NOT_WORK_REGION:   toBoolean(replace(replace(row.
  REG_REGION_NOT_WORK_REGION, '1', 'TRUE'), '0', 'FALSE')),
                         LIVE_REGION_NOT_WORK_REGION:  toBoolean(replace(replace(row.
  LIVE_REGION_NOT_WORK_REGION, '1', 'TRUE'), '0', 'FALSE')),
                         REG_CITY_NOT_LIVE_CITY:       toBoolean(replace(replace(row.
  REG_CITY_NOT_LIVE_CITY, '1', 'TRUE'), '0', 'FALSE')),
                         REG_CITY_NOT_WORK_CITY:       toBoolean(replace(replace(row.
  REG_CITY_NOT_WORK_CITY, '1', 'TRUE'), '0', 'FALSE')),
                         LIVE_CITY_NOT_WORK_CITY:      toBoolean(replace(replace(row.
  LIVE_CITY_NOT_WORK_CITY, '1', 'TRUE'), '0', 'FALSE')),
                         ORGANIZATION_TYPE:            row.ORGANIZATION_TYPE,
                         EXT_SOURCE_1:                 toFloat(row.EXT_SOURCE_1),
                         EXT_SOURCE_2:                 toFloat(row.EXT_SOURCE_2),
                         EXT_SOURCE_3:                 toFloat(row.EXT_SOURCE_3),
                         APARTMENTS_AVG:               toFloat(row.APARTMENTS_AVG),
                         BASEMENTAREA_AVG:             toFloat(row.BASEMENTAREA_AVG),
                         YEARS_BEGINEXPLUATATION_AVG:  toFloat(row.YEARS_BEGINEXPLUATATION_AVG),
                         YEARS_BUILD_AVG:              toFloat(row.YEARS_BUILD_AVG),
                         COMMONAREA_AVG:               toFloat(row.COMMONAREA_AVG),
                         ELEVATORS_AVG:                toFloat(row.ELEVATORS_AVG),
                         ENTRANCES_AVG:                toFloat(row.ENTRANCES_AVG),
                         FLOORSMAX_AVG:                toFloat(row.FLOORSMAX_AVG),
                         FLOORSMIN_AVG:                toFloat(row.FLOORSMIN_AVG),
                         LANDAREA_AVG:                 toFloat(row.LANDAREA_AVG),
                         LIVINGAPARTMENTS_AVG:         toFloat(row.LIVINGAPARTMENTS_AVG),
                         LIVINGAREA_AVG:               toFloat(row.LIVINGAREA_AVG),
                         NONLIVINGAPARTMENTS_AVG:      toFloat(row.NONLIVINGAPARTMENTS_AVG),
                         NONLIVINGAREA_AVG:            toFloat(row.NONLIVINGAREA_AVG),
                         APARTMENTS_MODE:              toFloat(row.APARTMENTS_MODE),
                         BASEMENTAREA_MODE:            toFloat(row.BASEMENTAREA_MODE),
                         YEARS_BEGINEXPLUATATION_MODE: toFloat(row.YEARS_BEGINEXPLUATATION_MODE),
                         YEARS_BUILD_MODE:             toFloat(row.YEARS_BUILD_MODE),
                         COMMONAREA_MODE:              toFloat(row.COMMONAREA_MODE),
                         ELEVATORS_MODE:               toFloat(row.ELEVATORS_MODE),
                         ENTRANCES_MODE:               toFloat(row.ENTRANCES_MODE),
                         FLOORSMAX_MODE:               toFloat(row.FLOORSMAX_MODE),
                         FLOORSMIN_MODE:               toFloat(row.FLOORSMIN_MODE),
                         LANDAREA_MODE:                toFloat(row.LANDAREA_MODE),
                         LIVINGAPARTMENTS_MODE:        toFloat(row.LIVINGAPARTMENTS_MODE),
                         LIVINGAREA_MODE:              toFloat(row.LIVINGAREA_MODE),
                         NONLIVINGAPARTMENTS_MODE:     toFloat(row.NONLIVINGAPARTMENTS_MODE),
                         NONLIVINGAREA_MODE:           toFloat(row.NONLIVINGAREA_MODE),
                         APARTMENTS_MEDI:              toFloat(row.APARTMENTS_MEDI),
                         BASEMENTAREA_MEDI:            toFloat(row.BASEMENTAREA_MEDI),
                         YEARS_BEGINEXPLUATATION_MEDI: toFloat(row.YEARS_BEGINEXPLUATATION_MEDI),
                         YEARS_BUILD_MEDI:             toFloat(row.YEARS_BUILD_MEDI),
                         COMMONAREA_MEDI:              toFloat(row.COMMONAREA_MEDI),
                         ELEVATORS_MEDI:               toFloat(row.ELEVATORS_MEDI),
                         ENTRANCES_MEDI:               toFloat(row.ENTRANCES_MEDI),
                         FLOORSMAX_MEDI:               toFloat(row.FLOORSMAX_MEDI),
                         FLOORSMIN_MEDI:               toFloat(row.FLOORSMIN_MEDI),
                         LANDAREA_MEDI:                toFloat(row.LANDAREA_MEDI),
                         LIVINGAPARTMENTS_MEDI:        toFloat(row.LIVINGAPARTMENTS_MEDI),
                         LIVINGAREA_MEDI:              toFloat(row.LIVINGAREA_MEDI),
                         NONLIVINGAPARTMENTS_MEDI:     toFloat(row.NONLIVINGAPARTMENTS_MEDI),
                         NONLIVINGAREA_MEDI:           toFloat(row.NONLIVINGAREA_MEDI),
                         FONDKAPREMONT_MODE:           row.FONDKAPREMONT_MODE,
                         HOUSETYPE_MODE:               row.HOUSETYPE_MODE,
                         TOTALAREA_MODE:               toFloat(row.TOTALAREA_MODE),
                         WALLSMATERIAL_MODE:           row.WALLSMATERIAL_MODE,
                         EMERGENCYSTATE_MODE:          row.EMERGENCYSTATE_MODE,
                         OBS_30_CNT_SOCIAL_CIRCLE:     toFloat(row.OBS_30_CNT_SOCIAL_CIRCLE),
                         DEF_30_CNT_SOCIAL_CIRCLE:     toFloat(row.DEF_30_CNT_SOCIAL_CIRCLE),
                         OBS_60_CNT_SOCIAL_CIRCLE:     toFloat(row.OBS_60_CNT_SOCIAL_CIRCLE),
                         DEF_60_CNT_SOCIAL_CIRCLE:     toFloat(row.DEF_60_CNT_SOCIAL_CIRCLE),
                         DAYS_LAST_PHONE_CHANGE:       toFloat(row.DAYS_LAST_PHONE_CHANGE),
                         FLAG_DOCUMENT_2:              toBoolean(replace(replace(row.
  FLAG_DOCUMENT_2, '1', 'TRUE'), '0', 'FALSE')),
                         FLAG_DOCUMENT_3:              toBoolean(replace(replace(row.
  FLAG_DOCUMENT_3, '1', 'TRUE'), '0', 'FALSE')),
                         FLAG_DOCUMENT_4:              toBoolean(replace(replace(row.
  FLAG_DOCUMENT_4, '1', 'TRUE'), '0', 'FALSE')),
                         FLAG_DOCUMENT_5:              toBoolean(replace(replace(row.
  FLAG_DOCUMENT_5, '1', 'TRUE'), '0', 'FALSE')),
                         FLAG_DOCUMENT_6:              toBoolean(replace(replace(row.
  FLAG_DOCUMENT_6, '1', 'TRUE'), '0', 'FALSE')),
                         FLAG_DOCUMENT_7:              toBoolean(replace(replace(row.
  FLAG_DOCUMENT_7, '1', 'TRUE'), '0', 'FALSE')),
                         FLAG_DOCUMENT_8:              toBoolean(replace(replace(row.
  FLAG_DOCUMENT_8, '1', 'TRUE'), '0', 'FALSE')),
                         FLAG_DOCUMENT_9:              toBoolean(replace(replace(row.
  FLAG_DOCUMENT_9, '1', 'TRUE'), '0', 'FALSE')),
                         FLAG_DOCUMENT_10:             toBoolean(replace(replace(row.
  FLAG_DOCUMENT_10, '1', 'TRUE'), '0', 'FALSE')),
                         FLAG_DOCUMENT_11:             toBoolean(replace(replace(row.
  FLAG_DOCUMENT_11, '1', 'TRUE'), '0', 'FALSE')),
                         FLAG_DOCUMENT_12:             toBoolean(replace(replace(row.
  FLAG_DOCUMENT_12, '1', 'TRUE'), '0', 'FALSE')),
                         FLAG_DOCUMENT_13:             toBoolean(replace(replace(row.
  FLAG_DOCUMENT_13, '1', 'TRUE'), '0', 'FALSE')),
                         FLAG_DOCUMENT_14:             toBoolean(replace(replace(row.
  FLAG_DOCUMENT_14, '1', 'TRUE'), '0', 'FALSE')),
                         FLAG_DOCUMENT_15:             toBoolean(replace(replace(row.
  FLAG_DOCUMENT_15, '1', 'TRUE'), '0', 'FALSE')),
                         FLAG_DOCUMENT_16:             toBoolean(replace(replace(row.
  FLAG_DOCUMENT_16, '1', 'TRUE'), '0', 'FALSE')),
                         FLAG_DOCUMENT_17:             toBoolean(replace(replace(row.
  FLAG_DOCUMENT_17, '1', 'TRUE'), '0', 'FALSE')),
                         FLAG_DOCUMENT_18:             toBoolean(replace(replace(row.
  FLAG_DOCUMENT_18, '1', 'TRUE'), '0', 'FALSE')),
                         FLAG_DOCUMENT_19:             toBoolean(replace(replace(row.
  FLAG_DOCUMENT_19, '1', 'TRUE'), '0', 'FALSE')),
                         FLAG_DOCUMENT_20:             toBoolean(replace(replace(row.
  FLAG_DOCUMENT_20, '1', 'TRUE'), '0', 'FALSE')),
                         FLAG_DOCUMENT_21:             toBoolean(replace(replace(row.
  FLAG_DOCUMENT_21, '1', 'TRUE'), '0', 'FALSE')),
                         AMT_REQ_CREDIT_BUREAU_HOUR:   toFloat(row.AMT_REQ_CREDIT_BUREAU_HOUR),
                         AMT_REQ_CREDIT_BUREAU_DAY:    toFloat(row.AMT_REQ_CREDIT_BUREAU_DAY),
                         AMT_REQ_CREDIT_BUREAU_WEEK:   toFloat(row.AMT_REQ_CREDIT_BUREAU_WEEK),
                         AMT_REQ_CREDIT_BUREAU_MON:    toFloat(row.AMT_REQ_CREDIT_BUREAU_MON),
                         AMT_REQ_CREDIT_BUREAU_QRT:    toFloat(row.AMT_REQ_CREDIT_BUREAU_QRT),
                         AMT_REQ_CREDIT_BUREAU_YEAR:   toFloat(row.AMT_REQ_CREDIT_BUREAU_YEAR)
});

CALL db.awaitIndexes(600);

// Create Bureaus
USING PERIODIC COMMIT
LOAD CSV WITH HEADERS FROM 'file:///data/bureau.csv' AS row
CREATE (bur:Bureau {SK_ID_BUREAU:           toInteger(row.SK_ID_BUREAU),
                    CREDIT_ACTIVE:          row.CREDIT_ACTIVE,
                    CREDIT_CURRENCY:        row.CREDIT_CURRENCY,
                    DAYS_CREDIT:            toFloat(row.DAYS_CREDIT),
                    CREDIT_DAY_OVERDUE:     toFloat(row.CREDIT_DAY_OVERDUE),
                    DAYS_CREDIT_ENDDATE:    toFloat(row.DAYS_CREDIT_ENDDATE),
                    DAYS_ENDDATE_FACT:      toFloat(row.DAYS_ENDDATE_FACT),
                    AMT_CREDIT_MAX_OVERDUE: toFloat(row.AMT_CREDIT_MAX_OVERDUE),
                    CNT_CREDIT_PROLONG:     toInteger(row.CNT_CREDIT_PROLONG),
                    AMT_CREDIT_SUM:         toFloat(row.AMT_CREDIT_SUM),
                    AMT_CREDIT_SUM_DEBT:    toFloat(row.AMT_CREDIT_SUM_DEBT),
                    AMT_CREDIT_SUM_LIMIT:   toFloat(row.AMT_CREDIT_SUM_LIMIT),
                    AMT_CREDIT_SUM_OVERDUE: toFloat(row.AMT_CREDIT_SUM_OVERDUE),
                    CREDIT_TYPE:            row.CREDIT_TYPE,
                    DAYS_CREDIT_UPDATE:     toFloat(row.DAYS_CREDIT_UPDATE),
                    AMT_ANNUITY:            toFloat(row.AMT_ANNUITY)
})
WITH bur, row
MATCH (app:Application {SK_ID_CURR: toInteger(row.SK_ID_CURR)})
USING INDEX app:Application(SK_ID_CURR)
CREATE (bur)-[:BUREAU_TO_APPLICATION]->(app);

CALL db.awaitIndexes(600);

// Create BureauBalances
USING PERIODIC COMMIT
LOAD CSV WITH HEADERS FROM 'file:///data/bureau_balance.csv' AS row
CREATE (bb:BureauBalance {
  MONTHS_BALANCE: toInteger(row.MONTHS_BALANCE),
  STATUS:         row.STATUS
})
WITH bb, row
MATCH (bur:Bureau {SK_ID_BUREAU: toInteger(row.SK_ID_BUREAU)})
USING INDEX bur:Bureau(SK_ID_BUREAU)
CREATE (bb)-[:BUREAUBALANCE_TO_BUREAU]->(bur);

// Create PreviousApplications
USING PERIODIC COMMIT
LOAD CSV WITH HEADERS FROM 'file:///data/previous_application.csv' AS row
CREATE (papp:PreviousApplication {
  SK_ID_PREV:                  toInteger(row.SK_ID_PREV),
  NAME_CONTRACT_TYPE:          row.NAME_CONTRACT_TYPE,
  AMT_ANNUITY:                 toFloat(row.AMT_ANNUITY),
  AMT_APPLICATION:             toFloat(row.AMT_APPLICATION),
  AMT_CREDIT:                  toFloat(row.AMT_CREDIT),
  AMT_DOWN_PAYMENT:            toFloat(row.AMT_DOWN_PAYMENT),
  AMT_GOODS_PRICE:             toFloat(row.AMT_GOODS_PRICE),
  WEEKDAY_APPR_PROCESS_START:  row.WEEKDAY_APPR_PROCESS_START,
  HOUR_APPR_PROCESS_START:     toInteger(row.HOUR_APPR_PROCESS_START),
  FLAG_LAST_APPL_PER_CONTRACT: toBoolean(replace(replace(row.FLAG_LAST_APPL_PER_CONTRACT, 'Y', 'TRUE'), 'N', 'FALSE')),
  NFLAG_LAST_APPL_IN_DAY:      toBoolean(replace(replace(row.NFLAG_LAST_APPL_IN_DAY, '1', 'TRUE'), '0', 'FALSE')),
  RATE_DOWN_PAYMENT:           toFloat(row.RATE_DOWN_PAYMENT),
  RATE_INTEREST_PRIMARY:       toFloat(row.RATE_INTEREST_PRIMARY),
  RATE_INTEREST_PRIVILEGED:    toFloat(row.RATE_INTEREST_PRIVILEGED),
  NAME_CASH_LOAN_PURPOSE:      row.NAME_CASH_LOAN_PURPOSE,
  NAME_CONTRACT_STATUS:        row.NAME_CONTRACT_STATUS,
  DAYS_DECISION:               toFloat(row.DAYS_DECISION),
  NAME_PAYMENT_TYPE:           row.NAME_PAYMENT_TYPE,
  CODE_REJECT_REASON:          row.CODE_REJECT_REASON,
  NAME_TYPE_SUITE:             row.NAME_TYPE_SUITE,
  NAME_CLIENT_TYPE:            row.NAME_CLIENT_TYPE,
  NAME_GOODS_CATEGORY:         row.NAME_GOODS_CATEGORY,
  NAME_PORTFOLIO:              row.NAME_PORTFOLIO,
  NAME_PRODUCT_TYPE:           row.NAME_PRODUCT_TYPE,
  CHANNEL_TYPE:                row.CHANNEL_TYPE,
  SELLERPLACE_AREA:            toFloat(replace(row.SELLERPLACE_AREA, '-1', 'null')), // -1 values are nulls
  NAME_SELLER_INDUSTRY:        row.NAME_SELLER_INDUSTRY,
  CNT_PAYMENT:                 toFloat(row.CNT_PAYMENT),
  NAME_YIELD_GROUP:            row.NAME_YIELD_GROUP,
  PRODUCT_COMBINATION:         row.PRODUCT_COMBINATION,
  DAYS_FIRST_DRAWING:          toFloat(replace(row.DAYS_FIRST_DRAWING, '365243', 'null')), // 365243 means null
  DAYS_FIRST_DUE:              toFloat(replace(row.DAYS_FIRST_DUE, '365243', 'null')), // 365243 means null
  DAYS_LAST_DUE_1ST_VERSION:   toFloat(replace(row.DAYS_LAST_DUE_1ST_VERSION, '365243', 'null')), // 365243 means null
  DAYS_LAST_DUE:               toFloat(replace(row.DAYS_LAST_DUE, '365243', 'null')), // 365243 means null
  DAYS_TERMINATION:            toFloat(replace(row.DAYS_TERMINATION, '365243', 'null')), // 365243 means null
  NFLAG_INSURED_ON_APPROVAL:   toBoolean(replace(replace(row.NFLAG_INSURED_ON_APPROVAL, '1.0', 'TRUE'), '0.0', 'FALSE'))
})
WITH papp, row
MATCH (app:Application {SK_ID_CURR: toInteger(row.SK_ID_CURR)})
USING INDEX app:Application(SK_ID_CURR)
CREATE (papp)-[:PREVIOUSAPPLICATION_TO_APPLICATION]->(app);

CALL db.awaitIndexes(600);

// Create CashBalances
USING PERIODIC COMMIT
LOAD CSV WITH HEADERS FROM 'file:///data/POS_CASH_balance.csv' AS row
CREATE (cash:CashBalance {
  MONTHS_BALANCE:        toInteger(row.MONTHS_BALANCE),
  CNT_INSTALMENT:        toFloat(row.CNT_INSTALMENT),
  CNT_INSTALMENT_FUTURE: toFloat(row.CNT_INSTALMENT_FUTURE),
  NAME_CONTRACT_STATUS:  row.NAME_CONTRACT_STATUS,
  SK_DPD:                toInteger(row.SK_DPD),
  SK_DPD_DEF:            toInteger(row.SK_DPD_DEF)
})
WITH cash, row
MATCH (app:Application {SK_ID_CURR: toInteger(row.SK_ID_CURR)})
USING INDEX app:Application(SK_ID_CURR)
CREATE (cash)-[:CASHBALANCE_TO_APPLICATION]->(app)
WITH cash, row
MATCH (papp:PreviousApplication {SK_ID_PREV: toInteger(row.SK_ID_PREV)})
USING INDEX papp:PreviousApplication(SK_ID_PREV)
CREATE (cash)-[:CASHBALANCE_TO_PREVIOUSAPPLICATION]->(papp);

// Create CreditBalances
USING PERIODIC COMMIT
LOAD CSV WITH HEADERS FROM 'file:///data/credit_card_balance.csv' AS row
CREATE (credit:CreditBalance {
  MONTHS_BALANCE:             toInteger(row.MONTHS_BALANCE),
  AMT_BALANCE:                toFloat(row.AMT_BALANCE),
  AMT_CREDIT_LIMIT_ACTUAL:    toFloat(row.AMT_CREDIT_LIMIT_ACTUAL),
  AMT_DRAWINGS_ATM_CURRENT:   toFloat(row.AMT_DRAWINGS_ATM_CURRENT),
  AMT_DRAWINGS_CURRENT:       toFloat(row.AMT_DRAWINGS_CURRENT),
  AMT_DRAWINGS_OTHER_CURRENT: toFloat(row.AMT_DRAWINGS_OTHER_CURRENT),
  AMT_DRAWINGS_POS_CURRENT:   toFloat(row.AMT_DRAWINGS_POS_CURRENT),
  AMT_INST_MIN_REGULARITY:    toFloat(row.AMT_INST_MIN_REGULARITY),
  AMT_PAYMENT_CURRENT:        toFloat(row.AMT_PAYMENT_CURRENT),
  AMT_PAYMENT_TOTAL_CURRENT:  toFloat(row.AMT_PAYMENT_TOTAL_CURRENT),
  AMT_RECEIVABLE_PRINCIPAL:   toFloat(row.AMT_RECEIVABLE_PRINCIPAL),
  AMT_RECIVABLE:              toFloat(row.AMT_RECIVABLE),
  AMT_TOTAL_RECEIVABLE:       toFloat(row.AMT_TOTAL_RECEIVABLE),
  CNT_DRAWINGS_ATM_CURRENT:   toFloat(row.CNT_DRAWINGS_ATM_CURRENT),
  CNT_DRAWINGS_CURRENT:       toFloat(row.CNT_DRAWINGS_CURRENT),
  CNT_DRAWINGS_OTHER_CURRENT: toFloat(row.CNT_DRAWINGS_OTHER_CURRENT),
  CNT_DRAWINGS_POS_CURRENT:   toFloat(row.CNT_DRAWINGS_POS_CURRENT),
  CNT_INSTALMENT_MATURE_CUM:  toFloat(row.CNT_INSTALMENT_MATURE_CUM),
  NAME_CONTRACT_STATUS:       row.NAME_CONTRACT_STATUS,
  SK_DPD:                     toFloat(row.SK_DPD),
  SK_DPD_DEF:                 toFloat(row.SK_DPD_DEF)
})
WITH credit, row
MATCH (app:Application {SK_ID_CURR: toInteger(row.SK_ID_CURR)})
USING INDEX app:Application(SK_ID_CURR)
CREATE (credit)-[:CREDITBALANCE_TO_APPLICATION]->(app)
WITH credit, row
MATCH (papp:PreviousApplication {SK_ID_PREV: toInteger(row.SK_ID_PREV)})
USING INDEX papp:PreviousApplication(SK_ID_PREV)
CREATE (credit)-[:CREDITBALANCE_TO_PREVIOUSAPPLICATION]->(papp);

// Create InstallmentsPayments
USING PERIODIC COMMIT
LOAD CSV WITH HEADERS FROM 'file:///data/installments_payments.csv' AS row
CREATE (ip:InstallmentPayment {
  NUM_INSTALMENT_VERSION: toInteger(row.NUM_INSTALMENT_VERSION),
  NUM_INSTALMENT_NUMBER:  toInteger(row.NUM_INSTALMENT_NUMBER),
  DAYS_INSTALMENT:        toFloat(row.DAYS_INSTALMENT),
  DAYS_ENTRY_PAYMENT:     toFloat(row.DAYS_ENTRY_PAYMENT),
  AMT_INSTALMENT:         toFloat(row.AMT_INSTALMENT),
  AMT_PAYMENT:            toFloat(row.AMT_PAYMENT)
})
WITH ip, row
MATCH (app:Application {SK_ID_CURR: toInteger(row.SK_ID_CURR)})
USING INDEX app:Application(SK_ID_CURR)
CREATE (ip)-[:INSTALLMENTPAYMENT_TO_APPLICATION]->(app)
WITH ip, row
MATCH (papp:PreviousApplication {SK_ID_PREV: toInteger(row.SK_ID_PREV)})
USING INDEX papp:PreviousApplication(SK_ID_PREV)
CREATE (ip)-[:INSTALLMENTPAYMENT_TO_PREVIOUSAPPLICATION]->(papp);