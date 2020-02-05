CREATE CONSTRAINT ON (proj:Project) ASSERT proj.project_id IS UNIQUE;

CREATE CONSTRAINT ON (res_type:ResourceType) ASSERT res_type.resource_type IS UNIQUE;


// Create Projects, and Resource Types
USING PERIODIC COMMIT
LOAD CSV WITH HEADERS FROM 'file:///data/projects.csv' AS row
CREATE (proj:Project {project_id:                             row.projectid,
                      school_id:                              row.schoolid,
                      school_ncesid:                          toInteger(row.school_ncesid),
                      school_latitude:                        toFloat(row.school_latitude),
                      school_longitude:                       toFloat(row.school_longitude),
                      school_city:                            row.school_city,
                      school_state:                           row.school_state,
                      school_zip:                             toInteger(row.school_zip),
                      school_metro:                           row.school_metro,
                      school_district:                        row.school_district,
                      school_county:                          row.school_county,
                      school_charter:                         toBoolean(replace(replace(row.
  school_charter, 't', 'TRUE'), 'f', 'FALSE')),
                      school_magnet:                          toBoolean(replace(replace(row.
  school_magnet, 't', 'TRUE'), 'f', 'FALSE')),
                      school_year_round:                      toBoolean(replace(replace(row.
  school_year_round, 't', 'TRUE'), 'f', 'FALSE')),
                      school_nlns:                            toBoolean(replace(replace(row.
  school_nlns, 't', 'TRUE'), 'f', 'FALSE')),
                      school_kipp:                            toBoolean(replace(replace(row.
  school_kipp, 't', 'TRUE'), 'f', 'FALSE')),
                      school_charter_ready_promise:           toBoolean(replace(replace(row.
  school_charter_ready_promise, 't', 'TRUE'), 'f', 'FALSE')),
                      teacher_prefix:                         row.teacher_prefix,
                      teacher_teach_for_america:              toBoolean(replace(replace(row.
  teacher_teach_for_america, 't', 'TRUE'), 'f', 'FALSE')),
                      teacher_ny_teaching_fellow:             toBoolean(replace(replace(row.
  teacher_ny_teaching_fellow, 't', 'TRUE'), 'f', 'FALSE')),
                      primary_focus_subject:                  row.primary_focus_subject,
                      primary_focus_area:                     row.primary_focus_area,
                      secondary_focus_subject:                row.secondary_focus_subject,
                      secondary_focus_area:                   row.secondary_focus_area,
                      poverty_level:                          row.poverty_level,
                      grade_level:                            row.grade_level,
                      fulfillment_labor_materials:            toFloat(row.fulfillment_labor_materials),
                      total_price_excluding_optional_support: toFloat(row.total_price_excluding_optional_support),
                      total_price_including_optional_support: toFloat(row.total_price_including_optional_support),
                      students_reached:                       toFloat(row.students_reached),
                      eligible_double_your_impact_match:      toBoolean(replace(replace(row.
  eligible_double_your_impact_match, 't', 'TRUE'), 'f', 'FALSE')),
                      eligible_almost_home_match:             toBoolean(replace(replace(row.
  eligible_almost_home_match, 't', 'TRUE'), 'f', 'FALSE')),
                      date_posted:                            date(row.date_posted),
                      is_exciting:                            null}
       )
WITH proj, row
  WHERE NOT row.resource_type IS NULL
MERGE (res_type:ResourceType {resource_type: row.resource_type})
CREATE (proj)-[:PROJECT_TO_RESOURCETYPE]->(res_type);

CALL db.awaitIndexes(600);

// Create Essays
USING PERIODIC COMMIT
LOAD CSV WITH HEADERS FROM 'file:///data/essays.csv' AS row
CREATE (ess:Essay {title:             row.title,
                   short_description: row.short_description,
                   need_statement:    row.need_statement,
                   essay:             row.essay})
WITH ess, row
MATCH (proj:Project {project_id: row.projectid})
USING INDEX proj:Project(project_id)
CREATE (ess)-[:ESSAY_TO_PROJECT]->(proj);

// Create Resources
USING PERIODIC COMMIT
LOAD CSV WITH HEADERS FROM 'file:///data/resources.csv' AS row
CREATE (res:Resource {vendor_id:       row.vendorid,
                      vendor_name:     row.vendor_name,
                      item_name:       row.item_name,
                      item_number:     row.item_number,
                      item_unit_price: toFloat(row.item_unit_price),
                      item_quantity:   toInteger(row.item_quantity)})
WITH res, row
MATCH (proj:Project {project_id: row.projectid})
USING INDEX proj:Project(project_id)
CREATE (res)-[:RESOURCE_TO_PROJECT]->(proj)
WITH res, row
MATCH (res_type:ResourceType {resource_type: row.project_resource_type})
USING INDEX res_type:ResourceType(resource_type)
CREATE (res)-[:RESOURCE_TO_RESOURCETYPE]->(res_type);

// Add training labels to Projects
USING PERIODIC COMMIT
LOAD CSV WITH HEADERS FROM 'file:///data/outcomes.csv' AS row
WITH row
MATCH (proj:Project {project_id: row.projectid})
SET proj.is_exciting = toBoolean(replace(replace(row.is_exciting, 't', 'TRUE'), 'f', 'FALSE'));


CALL db.awaitIndexes(600);
