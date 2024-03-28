#!/usr/bin/env python
# coding: utf-8

# ## Assignment 3 – Database Queries
# ### Introduction
# In this assignment you will write some simple SQL database queries, written in Python, and using sqlite. 
# 
# Ensure you have read the entire description on the course page before starting this notebook file.
# 
# ### Testing
# Each of the tasks below has a sample test to help you develop your solution. You are encouraged to write additional tests yourself. Your assignment will be graded based on hidden tests for each task.
# 
# Some of the tasks modify the database, and the provided tests have been written in a way that assumes the entire notebook has been run in order. Always re-run the entire notebook when you test your code, and you may wish to restart the kernel too.
# 
# The hidden tests will always construct the database afresh with correct data, meaning that even if you make a mistake in, say, task 4, you can still get full marks for tasks 5 onward. Note that the provided tests may *not* work in these conditions.
# 
# ### Utility functions
# The functions below are used to provide a simple SQL interface with sqlite. You should look at what they do but should not need to change them.

# In[1]:


# This code wraps up the database access in a single function.
import sqlite3

# access_database requires the name of a sqlite3 database file, the query, and its parameters.
# It does not return the result of the query.
def access_database(dbfile, query, parameters=()):
    connect = sqlite3.connect(dbfile)
    cursor = connect.cursor()
    cursor.execute(query, parameters)
    connect.commit()
    connect.close()

# access_database requires the name of a sqlite3 database file, the query, and its parameters.
# It returns the result of the query
def access_database_with_result(dbfile, query, parameters=()):
    connect = sqlite3.connect(dbfile)
    cursor = connect.cursor()
    rows = cursor.execute(query, parameters).fetchall()
    connect.commit()
    connect.close()
    return rows


# ### Database design 
# Here is a list of tables, along with detailed explanations of the columns in each table.
# 
# ##### Units
# This table holds the basic detail of a unit.
# 
# * `UnitID` A unique integer identifier used to reference a unit.
# * `Name` The string name of the unit.
# * `Level` The educational level of the unit.
# * `Semester` The semester in which the unit occurs, where 0 indicates the whole year.
# 
# ##### Students
# This table holds the basic detail of a student.
# 
# * `StudentID` A unique integer identifier used to reference a student.
# * `Name` A string name of the student.
# 
# ##### Enrolled
# This table indicates which students are enrolled on a unit and in which year. Note that it uses the ID values of the student and unit to create the relationship.
# 
# * `StudentID` The ID of a student.
# * `UnitID` The ID of a unit.
# * `Year` An integer indicate the year the student was enrolled in the unit.
# 
# ##### Assessments
# An assessment is an piece of work created by the lecturer for a unit, in a given year. The weightings for a unit in a given year are expected to sum to 100%
# 
# * `AssessmentID` A unique integer identifier used to reference an assignment.
# * `UnitID` Which unit this assessment belongs to.
# * `Year` The integer year that this assignment relates to.
# * `Name` A string name for the assessment.
# * `Mark` The integer maximum mark available for the assignment.
# * `Weighting` The integer weighting out of 100 for this assignment.
# * `Deadline` The deadline set for this assignment.
# 
# ##### Assignments
# An assignment is the record of the assessment for an individual student.
# 
# * `AssignmentID` A unique integer identifier used to reference an assignment instance.
# * `StudentID` The student this instance relates to.
# * `AssessmentID` The assessment this instance relates to.
# * `Deadline` The deadline for this student.
# * `Submitted` The date on which this assessment was submitted.
# * `Mark` The mark allocated to this piece of work.
# * `Marked` A flag indicating if this piece of work has been marked. 0:no, 1: yes.
# 
# ### Database setup
# The following function initialises (or reinitialises) the database, adding the tables as specified above, and inserting some sample data.
# 
# You should not change the code below since the hidden tests will use the same database structure, and modifications to the code below may not propagate to the hidden tests.

# In[2]:


# Set up the tables

def setup_assessment_tables(dbfile):
    # Get rid of any existing data
    access_database(dbfile, "DROP TABLE IF EXISTS Units")
    access_database(dbfile, "DROP TABLE IF EXISTS Students")
    access_database(dbfile, "DROP TABLE IF EXISTS Enrolled")
    access_database(dbfile, "DROP TABLE IF EXISTS Assessments")
    access_database(dbfile, "DROP TABLE IF EXISTS Assignments")

    # Freshly setup tables
    access_database(dbfile, "CREATE TABLE Units (UnitID INT, Name TEXT, Level INT, Semester INT)")
    access_database(dbfile, "CREATE TABLE Students (StudentID INT, Name TEXT)")
    access_database(dbfile, "CREATE TABLE Enrolled (StudentID INT, UnitID INT, Year INT)")
    access_database(dbfile, "CREATE TABLE Assessments (AssessmentID INT, UnitID INT, Year INT, Name TEXT, Mark INT, Weighting INT, Deadline DATE)")
    access_database(dbfile, "CREATE TABLE Assignments (AssignmentID INTEGER PRIMARY KEY AUTOINCREMENT, StudentID INT, AssessmentID INT, Deadline DATE, Submitted DATE, Mark INT, Marked INT)")

    # Populate the tables with some initial data
    access_database(dbfile, "INSERT INTO Units VALUES (100,'CM60100', 6, 1), (101,'CM60101', 6, 1), (102,'XX60200', 6, 1)")
    access_database(dbfile, "INSERT INTO Students VALUES (1001,'Rod'),(1002,'Jane'),(1003,'Freddy')")
    access_database(dbfile, "INSERT INTO Enrolled VALUES (1001,100,2020), (1001,101,2020), (1002,100,2019), (1002,101,2020), (1002,102,2019), (1003, 101, 2019), (1003, 102, 2019)")
    access_database(dbfile, "INSERT INTO Assessments VALUES (1,100,2020,'Exam',100,75,'2021-1-25 20:00'), (2,100,2020,'Coursework',100,25,'2020-12-25 20:00'), (3,101,2020,'Coursework',50,100,'2020-12-15 20:00'), (4,102,2019,'Coursework',50,100,'2019-12-15 20:00')")


# In[3]:


# To test your code, always re-run the entire notebook (since tests may change the database)
setup_assessment_tables("database.db")


# ### Tasks
# In each of the following tasks, you will need to write one or more SQL queries to complete a Python function. 
# 
# #### Important: Security note
# When you construct SQL queries within code, your first thought might be to do something like this
# 
# ```python
# def student_details(studentid):
#     query = "SELECT * FROM Students WHERE StudentID == " + str(studentid) + ";"
#     db.execute(query)
#     ...
# ```
# 
# But this leaves your code susceptible to an SQL injection attack. Rather than a valid ID, a user of your function could insert a string into your function such as [`"1; DROP TABLE Students;"`](https://xkcd.com/327/), and you would unwittingly run the query provided – in this case, deleting the Students table. You can read more [online](https://en.wikipedia.org/wiki/SQL_injection).
# 
# This is why SQLite provides a method for *parameter substitution*. In the query you write single question mark `?`, and then provide a *tuple* of values to the `execute` function, like so:
# 
# ```python
# def student_details(studentid):
#     query = "SELECT * FROM Students WHERE StudentID == ?;"
#     db.execute(query, (studentid,))
#     ...
# ```
# 
# The helper functions `access_database` and `access_database_with_result` mirror this, and this is demonstrated below. It is a good habit to get into the practice of providing parameters this way. 
# 
# #### Example
# Here is an example which demonstrates a function that returns all student details for a given student ID, as above.

# In[4]:


def student_details(database, studentid):
    row = access_database_with_result(database, "SELECT * FROM Students WHERE StudentID == ?;", (studentid,))
    return row

print(student_details("database.db", 1001))


# #### Task 1 (1 Mark)
# Provide a function that indicates the units a student is taking in a given year. 
# 
# It should return a list of the units being taken in increasing order of UnitID. e.g. `[101,104,105]`

# In[5]:


def student_units(database, studentid, year):

    query = """
                SELECT UnitID 
                FROM Enrolled 
                WHERE StudentID == ?
                AND Year == ?
                ORDER BY UnitID asc
                
            """
    
    row = access_database_with_result(database, query, (studentid, year))
    
    return [entry[0] for entry in row]


# In[6]:


result = student_units("database.db",1001,2020)
assert result == [100, 101]


# #### Task 2 (1 Mark): 
# Provide a function that indicates the students that are enrolled on a unit.
# 
# It should return a list of tuples of StudentID and Name ordered by StudentID. e.g. `[(110,'Zipppy'),(111,'Bungle')]`
# 

# In[7]:


def unit_students(database, unitid, year):

    query = """
                SELECT Enrolled.StudentID, Students.Name
                FROM Students
                LEFT JOIN Enrolled ON Students.StudentID = Enrolled.StudentID
                WHERE Enrolled.UnitID == ? 
                AND Enrolled.Year == ?
                ORDER BY Enrolled.StudentID asc
                
            """
    
    row = access_database_with_result(database, query, (unitid, year))

    return row


# In[8]:


result = unit_students("database.db",101,2020)
assert result == [(1001, 'Rod'), (1002, 'Jane')]


# #### Task 3 (1 Marks): 
# Provide a function that indicates how many students are taking each unit in a given year.
# 
# It should return a list of tuples of UnitID, UnitName and Count ordered by UnitID. e.g. `[(1010,'Machine Learning',50), (1020,'Dissertaton',37)]`

# In[9]:


def unit_numbers(database, year):
    
    query = """
                SELECT Units.UnitID, Units.Name, count(Units.Name)
                FROM Units 
                LEFT JOIN Enrolled ON Enrolled.UnitID = Units.UnitID
                WHERE Enrolled.Year =  ?
                GROUP BY Units.Name
                ORDER BY Units.UnitID asc
                
            """
    
    row = access_database_with_result(database, query, (year,))

    return row


# In[10]:


result = unit_numbers("database.db",2020)
assert result == [(100, 'CM60100', 1), (101, 'CM60101', 2)]


# #### Task 4 (2 Marks): 
# Provide a function that uses the enrolments and assessments tables to fully populate the assignments table.
# 
# All assignments for each student in a given year should be created. Only the assignments a student should be undertaking should be created.

# In[11]:


def create_assignments(database, year):

    query = """
                SELECT Enrolled.StudentID, Assessments.AssessmentID, Assessments.Deadline
                FROM Assessments 
                LEFT JOIN Enrolled ON Enrolled.UnitID = Assessments.UnitID 
                WHERE Enrolled.Year = ?
                
            """
    
    rows = access_database_with_result(database, query, (year,))
    new_rows = [(i + 1, row[0], row[1], row[2], None, None, 0) for i, row in enumerate(rows)]

    insert_query = """
                        INSERT INTO Assignments (AssignmentID, StudentID, AssessmentID, Deadline, Submitted, Mark, Marked)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        
                   """

    for row in new_rows:
        access_database(database, insert_query, row)


# In[12]:


create_assignments("database.db",2020)
result = access_database_with_result("database.db","SELECT * FROM Assignments;")
assert result == [(1, 1001, 1, '2021-1-25 20:00', None, None, 0), 
                  (2, 1001, 2, '2020-12-25 20:00', None, None, 0), 
                  (3, 1001, 3, '2020-12-15 20:00', None, None, 0), 
                  (4, 1002, 3, '2020-12-15 20:00', None, None, 0)]


# #### Task 5 (1 Mark): 
# Update the mark of an assignment, given the StudentID, AssessmentID and Mark.
# 
# It should update the Marked flag and the Mark itself.

# In[13]:


def mark_assignment(database, studentid, assessmentid, mark):

    query = """
                UPDATE Assignments
                SET Mark = ?, Marked = 1
                WHERE StudentID = ? 
                AND AssessmentID = ? 
                
            """
    
    access_database(database, query, (mark, studentid, assessmentid))


# In[14]:


mark_assignment("database.db", 1001, 1, 57);
mark_assignment("database.db", 1001, 2, 11);
mark_assignment("database.db", 1001, 3, 45);
mark_assignment("database.db", 1002, 3, 40);
result = access_database_with_result("database.db","SELECT * FROM Assignments;")
assert result == [(1, 1001, 1, '2021-1-25 20:00', None, 57, 1), 
                  (2, 1001, 2, '2020-12-25 20:00', None, 11, 1), 
                  (3, 1001, 3, '2020-12-15 20:00', None, 45, 1), 
                  (4, 1002, 3, '2020-12-15 20:00', None, 40, 1)]


# #### Task 6 (2 Marks): 
# Compute the overall mark for all students taking a specified unit in a given year.

# In[15]:


def unit_marks(database, unitid, year):

    query = """
                SELECT Assignments.StudentID, ROUND(SUM((Assignments.Mark * 1.0 / Assessments.Mark) * Assessments.Weighting), 1)
                FROM Assessments 
                LEFT JOIN Assignments ON Assignments.AssessmentID = Assessments.AssessmentID 
                AND Assignments.Marked = 1 
                WHERE Assessments.UnitID = ? 
                AND Assessments.Year = ?
                GROUP BY Assignments.StudentID
                
            """
    
    rows = access_database_with_result("database.db", query, (unitid, year,))
    return rows


# In[16]:


result = unit_marks("database.db", 101, 2020)
assert result == [(1001, 90.0), (1002, 80.0)]


# #### Task 7 (2 Marks): 
# Compute the overall marks for each unit taken by a given student across all years.

# In[17]:


def student_marks(database, studentid):
        
    query = """
                    SELECT Assessments.UnitID, Assessments.Year, ROUND(SUM((Assignments.Mark * 1.0 / Assessments.Mark) * Assessments.Weighting), 1)
                    FROM Assessments
                    LEFT JOIN Assignments ON Assignments.AssessmentID = Assessments.AssessmentID
                    WHERE Assignments.StudentID = ?
                    GROUP BY Assessments.UnitID

              """

    rows = access_database_with_result(database, query, (studentid,))

    return rows


# In[18]:


result = student_marks("database.db", 1001)
assert result == [(100, 2020, 45.5), (101, 2020, 90.0)]


# In[19]:


# This is a test cell, do not delete or edit.

