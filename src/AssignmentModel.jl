__precompile__()
module AssignmentModel

using JuMP
using Cbc


export solve_model

"""
Solve model
cost = rand(1000,1000)
output = solve_model(cost)
"""
function solve_model(cost)

    solver = CbcSolver()
    m = Model(solver=solver)
    dim = size(cost)[1]

    @variable(m, x[1:dim, 1:dim]>=0)

    @constraints m begin
        # Constraint 1 - Only one value appears in each cell
        # Constraint 2 - Each value appears in each row once only
        # Constraint 3 - Each value appears in each column once only
        col[i=1:dim], sum(x[i,:]) == 1
        row[j=1:dim], sum(x[:,j]) == 1
    end

    @objective(m, Min, -sum(cost[i,j] * x[i,j] for i=1:dim, j=1:dim))

    # Solve it
    status = solve(m)

    # Check solution
    if status == :Infeasible
        error("No solution found!")
    else
        mipSol = getvalue(x)
        sol = zeros(Int,dim,dim)
        for row in 1:dim, col in 1:dim
            if mipSol[row, col] >= 0.9
                sol[row, col] = 1
            end
        end
        return sol
    end

end




end
