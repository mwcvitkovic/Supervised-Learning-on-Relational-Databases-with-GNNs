FROM neo4j:3.5.4

# Remove username and password requirement from browser and cypher-shell
RUN sed -i "s|#dbms.security.auth_enabled=false|dbms.security.auth_enabled=false|g" "${NEO4J_HOME}"/conf/neo4j.conf

# Change neo4j to use the container's root dir as its root dir, instead of some other one
RUN sed -i "s|dbms.directories.import=import|#dbms.directories.import=import|g" "${NEO4J_HOME}"/conf/neo4j.conf

# manual data load command:
# docker exec -i <container name> cypher-shell < <thing to pipe>