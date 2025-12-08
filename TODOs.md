# TODOs

## General
- Should we replace `linalg.inv` with transpose?
- Dataclass for component connection matrices?
- Variable/State descriptions
    - Can we put this in the globally scoped atrrs of each class?
- Can we create a silent mode? This would be helpful for testing
- When developing a new generator model, the name of the state is not reported.

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

## Paul Subtasks
1. Develop:
    - Transfer more GFMI and GFLI model from a previous STING implementation.
    - EMT modeling.
2. Test: 
    - GFLI and GFMI models.