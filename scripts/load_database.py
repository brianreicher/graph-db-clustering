import graphnn

if __name__ == "__main__":
    print("initialize driver")
    database: graphnn.imageDB.ImageDB = graphnn.imageDB.ImageDB("neo4j://localhost:7687", "neo4j", "password")
    database.connect()
    print("driver working")

    # drop existing database data
    database.flush_database()
    print("Data flushed")

    # fill the db with spotify csv data
    database.setSchema()
    print("Data added")