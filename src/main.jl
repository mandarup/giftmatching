
using DataStructures
using DataFrames
using MLDataUtils
using CSV
using JuMP
using JLD
using ProgressMeter
using PyPlot

include("Constants.jl")
# include("src/Constants.jl")
using Constants

# include("src/AssignmentModel.jl")
# include("src/Utils.jl")
# include("src/Heuristics.jl")
# include("src/Preprocess.jl")

include("AssignmentModel.jl")
include("Utils.jl")
include("Heuristics.jl")
include("Preprocess.jl")

using AssignmentModel
using Utils
using Heuristics
using Preprocess


BLOCK_SIZE = 100
N_ROUNDS = 50


# gift_happiness, child_happiness = Preprocess.load_happiness()
#
# sol = Heuristics.heuristic_dumb()
# info("solution feaible: ", Utils.check_feas(sol))
#
#
# slice_kids = rand(1:N_CHILDREN,BLOCK_SIZE)
# slice_kids= get_feasible_slice()
# solution_block = sol[slice_kids+1, :]
# # child_block = solution_block[:,1]
# # gift_block = solution_block[:,2]
#
#
# triplets = get_triplet_index(slice_kids)
# twins = get_twin_index(slice_kids)
# kids = solution_block[:,1]
# gifts = solution_block[:,2]




function optimize_block(solution_block)
    C = zeros((BLOCK_SIZE, BLOCK_SIZE))

    child_block = solution_block[:,1]
    gift_block = solution_block[:,2]

    for i in 1:BLOCK_SIZE
        c = child_block[i]
        # @show c
        for j in 1:BLOCK_SIZE
            # @show j, gift_block[j]
            g = gift_block[j]
            # @show c, g
            C[i, j] = child_happiness[c][g] + gift_happiness[g][c]
        end
    end
    # C

    assignment = AssignmentModel.solve_model(C)

    c, g = findn(assignment .== 1)

    kids, gifts = child_block[c], gift_block[g]
    return kids, gifts
end


function optimize(kids, gifts, child_happiness, gift_happiness; triplets=[], twins=[])

    block_size = length(kids)
    C = zeros((block_size, block_size))

    # child_block = solution_block[:,1]
    # gift_block = solution_block[:,2]

    for i in 1:block_size
        c = kids[i]
        # @show c
        for j in 1:block_size
            # @show j, gift_block[j]
            g = gifts[j]
            # @show c, g
            C[i, j] = child_happiness[c][g] + gift_happiness[g][c]
        end
    end

    assignment = AssignmentModel.solve_model(C, gifts; triplets=triplets, twins=twins)

    # sum(assignment .* C)

    c, g = findn(assignment .== 1)

    opt_kids, opt_gifts = kids[c], gifts[g]
    return opt_kids, opt_gifts
end



function compute_happiness(sol, child_happiness, gift_happiness)
    gift_assignment = Dict(zip(sol[:,1] , sol[:,2]))
    gift_counts = Dict(v=>0 for (k,v) in gift_assignment )
    tic()
    for (kid, gift) in gift_assignment
        gift_counts[gift] += 1
        if gift_counts[gift] > N_GIFT_QUANTITY
            throw(AssertionError("quantity for $gift==$(gift_counts[gift]) >=$N_GIFT_QUANTITY"))
        end
    end
    println("checking feasibility: ", Utils.check_feas(sol))

    max_child_happiness = N_GIFT_PREF * RATIO_CHILD_HAPPINESS
    max_gift_happiness = N_CHILD_PREF * RATIO_GIFT_HAPPINESS

    total_child_happiness = 0
    total_gift_happiness = 0 #zeros(N_GIFT_TYPE)
    # total_gift_happiness = zeros(N_GIFT_TYPE)

    for (c, g) in gift_assignment
        total_child_happiness +=  child_happiness[c][g]
        total_gift_happiness += gift_happiness[g][c]
    end
    nch = total_child_happiness / (N_CHILDREN * max_child_happiness)
    ngh = total_gift_happiness / (N_GIFT_QUANTITY * N_GIFT_TYPE * max_gift_happiness)
    println("normalized happiness child: $nch, gift: $ngh" )

    avg_happiness = nch ^3 + ngh ^3
    println("happiness: $avg_happiness")
    return avg_happiness
end

"""
If sampling triplets or twins, sample together. Add constraint to ensure same
gift.
"""
function get_feasible_slice(;sample_size=10)
    slice_kids = rand([TRIPLETS_INDEX; TWINS_INDEX; SINGLE_INDEX], sample_size)

    triplet_ids = intersect(slice_kids, TRIPLETS_INDEX)
    twin_ids = intersect(slice_kids, TWINS_INDEX)
    for i in triplet_ids
        index = findfirst(slice_kids, i)
        splice!(slice_kids, index, i:i+2)
        # append!(triplets_index, index:index+2)
    end
    for i in twin_ids
        index = findfirst(slice_kids, i)
        splice!(slice_kids, index, i:i+1)
        # append!(twins_index, index:index+1])
    end

    return  convert(Array{Int64,1},slice_kids)
end


function get_twin_index(slice_kids)
    twins_index = []
    twin_ids = intersect(slice_kids, TWINS_INDEX)
    for i in twin_ids
        index = findfirst(slice_kids, i)
        append!(twins_index, index)
    end

    return  twins_index
end

function get_triplet_index(slice_kids)
    triplets_index = []
    twins_index = []

    triplet_ids = intersect(slice_kids, TRIPLETS_INDEX)
    for i in triplet_ids
        index = findfirst(slice_kids, i)
        append!(triplets_index, index)
    end

    return  triplets_index
end





function run_opt(init, child_happiness, gift_happiness; history=[], n_rounds=10, sample_size=10)
    sol= copy(init)

    @showprogress 1 for i in 1:n_rounds
        # info("round $i")

        # slice_kids = rand(1:N_CHILDREN,BLOCK_SIZE)
        # slice_kids, triplets, twins = get_feasible_slice()
        slice_kids = get_feasible_slice(sample_size=sample_size)

        triplets = get_triplet_index(slice_kids)
        twins = get_twin_index(slice_kids)
        # @show twins
        # @show triplets

        solution_block = sol[slice_kids+1, :]
        current_kids = solution_block[:,1]
        current_gifts = solution_block[:,2]

        # info("optimizing")
        tic()
        # kids, gifts = optimize_block(solution_block)
        try
            kids, gifts = optimize(current_kids, current_gifts, child_happiness, gift_happiness; triplets=triplets, twins=twins)
            # println("optimization time :", toc())

            sol_copy = copy(sol)
            sol_copy[kids+1,2] = gifts
            if Utils.check_feas(sol_copy) == true
                sol =  sol_copy
            else
                info("new solution infeasible")
            end
        catch e
            @show e
        end
        compute_happiness(sol, child_happiness, gift_happiness)
    end
    return sol
end



function main()
    gift_pref, child_pref, gift_happiness, child_happiness = Preprocess.load_data()

    history = fill!(Array{Float64}(1),0)
    sol = Heuristics.heuristic_greedy(gift_pref)
    Utils.check_feas(sol)
    happiness = compute_happiness(sol, child_happiness, gift_happiness)
    append!(history, happiness)

    for i=1:100
        sol = run_opt(sol, child_happiness, gift_happiness; history=history, n_rounds=10, sample_size=30)
        happiness = compute_happiness(sol, child_happiness, gift_happiness)
        append!(history, happiness)
        println("happiness $(history[end-1]) -> $(history[end])")
        println("happiness gain: $(history[end] - history[end-1])")

        if happiness[end] < happiness[end-1]
            break
        end
        output = convert(DataFrame, sol)
        names!(output, [:ChildId, :GiftId])
        CSV.write(joinpath(pwd(),"data/output/sub_$(round(history[end],2)).csv"),output)
    end
end



main()


# Utils.avg_normalized_happiness(sol, child_pref, gift_pref)
# plot(history)

# write solution
# output = convert(DataFrame, sol)
# names!(output, [:ChildId, :GiftId])
# CSV.write(joinpath(pwd(),"data/output/sub_$(round(history[end],2)).csv"),output)



## load initial solution
# Utils.avg_normalized_happiness(sol, child_pref, gift_pref)
# init_df = CSV.read(joinpath(pwd(),"data/output/cpp_sub.csv"), header=true, nullable=false)
# init = convert(Array, init_df[2:end, :])
#
# sol = copy(init)
# Utils.check_feas(sol)
