# TODOs

## General
- Update export to matlab in the system class? 
- Should we replace `linalg.inv` with transpose?
- Dataclass for component connection matrices?
- Variable/State descriptions
    - Can we put this in the globally scoped atrrs of each class?
- Update gen models to new state-space model
- Update naming to camelcase classes 

- Note: Power flow uses strings indices for generators and buses. Maybe we should use int?

## Adam Subtasks
1. Testing
    - Add unit tests
        - StateSpaceModels
        - Variables
    - Validation
        - Update CSVs from MATLAB to Python
        - Define test inputs
        - Save EMT output files

2. Model reduction
    - Finish SSM class
    - Add gramians
    - Add intraconnect

3. Other
 - Finish ComponentConnections class


## Paul Subtasks