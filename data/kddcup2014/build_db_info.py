from data.utils import build_db_info

db_info = {'task': {'type': 'classification',
                    'n_classes': 2,
                    'n_train': 619326,
                    'n_test': 44772,
                    'train_class_counts': [582616, 36710]},
           'node_type_to_int': {
               'Project': 0,
               'Essay': 1,
               'Resource': 2,
               'ResourceType': 3
           },
           'edge_type_to_int': {
               'SELF': 0,
               'PROJECT_TO_RESOURCETYPE': 1,
               'ESSAY_TO_PROJECT': 2,
               'RESOURCE_TO_PROJECT': 3,
               'RESOURCE_TO_RESOURCETYPE': 4
           },
           'node_types_and_features': {'Project': {'school_id': {'type': 'CATEGORICAL'},
                                                   'school_ncesid': {'type': 'CATEGORICAL'},
                                                   'school_latitude+++school_longitude': {'type': 'LATLONG'},
                                                   'school_city': {'type': 'CATEGORICAL'},
                                                   'school_state': {'type': 'CATEGORICAL'},
                                                   'school_zip': {'type': 'CATEGORICAL'},
                                                   'school_metro': {'type': 'CATEGORICAL'},
                                                   'school_district': {'type': 'CATEGORICAL'},
                                                   'school_county': {'type': 'CATEGORICAL'},
                                                   'school_charter': {'type': 'CATEGORICAL'},
                                                   'school_magnet': {'type': 'CATEGORICAL'},
                                                   'school_year_round': {'type': 'CATEGORICAL'},
                                                   'school_nlns': {'type': 'CATEGORICAL'},
                                                   'school_kipp': {'type': 'CATEGORICAL'},
                                                   'school_charter_ready_promise': {'type': 'CATEGORICAL'},
                                                   'teacher_prefix': {'type': 'CATEGORICAL'},
                                                   'teacher_teach_for_america': {'type': 'CATEGORICAL'},
                                                   'teacher_ny_teaching_fellow': {'type': 'CATEGORICAL'},
                                                   'primary_focus_subject': {'type': 'CATEGORICAL'},
                                                   'primary_focus_area': {'type': 'CATEGORICAL'},
                                                   'secondary_focus_subject': {'type': 'CATEGORICAL'},
                                                   'secondary_focus_area': {'type': 'CATEGORICAL'},
                                                   'poverty_level': {'type': 'CATEGORICAL'},
                                                   'grade_level': {'type': 'CATEGORICAL'},
                                                   'fulfillment_labor_materials': {'type': 'SCALAR'},
                                                   'total_price_excluding_optional_support': {'type': 'SCALAR'},
                                                   'total_price_including_optional_support': {'type': 'SCALAR'},
                                                   'students_reached': {'type': 'SCALAR'},
                                                   'eligible_double_your_impact_match': {'type': 'CATEGORICAL'},
                                                   'eligible_almost_home_match': {'type': 'CATEGORICAL'},
                                                   'date_posted': {'type': 'DATETIME'},
                                                   'is_exciting': {'type': 'CATEGORICAL'},
                                                   },
                                       'Essay': {'title': {'type': 'TEXT'},
                                                 'short_description': {'type': 'TEXT'},
                                                 'need_statement': {'type': 'TEXT'},
                                                 'essay': {'type': 'TEXT'},
                                                 },
                                       'Resource': {'vendor_id': {'type': 'CATEGORICAL'},
                                                    'vendor_name': {'type': 'CATEGORICAL'},
                                                    'item_name': {'type': 'TEXT'},
                                                    'item_number': {'type': 'TEXT'},
                                                    'item_unit_price': {'type': 'SCALAR'},
                                                    'item_quantity': {'type': 'SCALAR'}
                                                    },
                                       'ResourceType': {'resource_type': {'type': 'CATEGORICAL'}
                                                        },
                                       },
           'label_feature': 'Project.is_exciting'
           }

if __name__ == '__main__':
    db_name = 'kddcup2014'
    test_dp_query = "MATCH (p:Project) WHERE p.date_posted >= date('2014-01-01') RETURN p.project_id ORDER BY p.project_id "
    train_dp_query = "MATCH (p:Project) WHERE p.date_posted < date('2014-01-01') RETURN p.project_id ORDER BY p.project_id "
    build_db_info(db_name, db_info, test_dp_query, train_dp_query)
