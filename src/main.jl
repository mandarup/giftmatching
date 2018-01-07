
using DataStructures
using DataFrames
using MLDataUtils
using CSV
using JuMP
using JLD
# using ProgressMeter
using ProgressMeter
using PyPlot
using Combinatorics
using StatsBase

include("Constants.jl")
# include("src/Constants.jl")
using Constants


#
# include("src/AssignmentModel.jl")
# include("src/Utils.jl")
# include("src/Heuristics.jl")
# include("src/Preprocess.jl")

include("AssignmentModel.jl")
include("Utils.jl")
include("Heuristics.jl")
include("Preprocess.jl")

# using AssignmentModel
# using Utils
# using Heuristics
# using Preprocess


function optimize(kids, gifts, child_happiness, gift_happiness; triplets=[], twins=[])
    block_size = length(kids)
    cost = zeros((block_size, block_size))

    for i in 1:block_size
        c = kids[i]
        # @show c
        for j in 1:block_size
            # @show j, gift_block[j]
            g = gifts[j]
            # @show c, g
            cost[i, j] = child_happiness[c][g] + gift_happiness[g][c]
        end
    end
    assignment = AssignmentModel.solve_model(cost, gifts; triplets=triplets, twins=twins)
    c, g = findn(assignment .== 1)
    opt_kids, opt_gifts = kids[c], gifts[g]
    return opt_kids, opt_gifts
end




function compute_happiness_vectors(sol, child_happiness, gift_happiness)
    num_rows = size(sol)[1]
    happiness = fill!(Array{Float64,2}(num_rows,2), 0)
    # for c in 0:num_rows-1
    #     g = sol[c+1,2]
    #     happiness[c+1,1] = child_happiness[c][g]
    #     happiness[c+1,2] = gift_happiness[g][c]
    # end

    for i in 1:num_rows
        g = sol[i,2]
        c = sol[i,1]
        happiness[i,1] = child_happiness[c][g]
        happiness[i,2] = gift_happiness[g][c]
    end
    return happiness
end


function compute_avg_happiness(happiness)
    nch = mean(happiness[:,1])
    ngh = mean(happiness[:,2])
    avg_happiness = nch ^3 + ngh ^3
    return avg_happiness
end


function compute_happiness_v3(sol, child_happiness, gift_happiness)
    happiness_vectors = compute_happiness_vectors(sol, child_happiness, gift_happiness)
    avg_happiness = compute_avg_happiness(happiness_vectors)
    return avg_happiness
end


function compute_happiness_v2(sol, child_happiness, gift_happiness)

    tic()
    gift_counts = fill!(Array{Int64,1}(N_GIFT_TYPE), 0) #Dict(v=>0 for (k,v) in gift_assignment )
    for kid in 0:N_CHILDREN-1
        gift = sol[kid+1,2]
        gift_counts[gift+1] += 1
        if gift_counts[gift+1] > N_GIFT_QUANTITY
            throw(AssertionError("quantity for $gift==$(gift_counts[gift]) >=$N_GIFT_QUANTITY"))
        end
    end
    toc()
    println("checking feasibility: ", Utils.check_feas(sol))

    total_child_happiness = 0
    total_gift_happiness = 0 #zeros(N_GIFT_TYPE)

    tic()
    for c in 0:N_CHILDREN-1
        g = sol[c+1,2]
        total_child_happiness +=  child_happiness[c][g]
        total_gift_happiness += gift_happiness[g][c]

    end


    nch = total_child_happiness / (N_CHILDREN )
    ngh = total_gift_happiness / (N_GIFT_QUANTITY * N_GIFT_TYPE )
    println("normalized happiness child: $nch, gift: $ngh" )

    avg_happiness = nch ^3 + ngh ^3
    println("happiness: $avg_happiness  ($(toc()))")
    toc()

    return avg_happiness
end


function compute_happiness_v1(sol, child_happiness, gift_happiness)
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

    # total_child_happiness = @parallel (+) for (c, g) in gift_assignment
    #     child_happiness[c][g]
    # end
    # total_gift_happiness = @parallel (+) for (c, g) in gift_assignment
    #     gift_happiness[g][c]
    # end


    nch = total_child_happiness / (N_CHILDREN )
    ngh = total_gift_happiness / (N_GIFT_QUANTITY * N_GIFT_TYPE )
    println("normalized happiness child: $nch, gift: $ngh" )

    avg_happiness = nch ^3 + ngh ^3
    println("happiness: $avg_happiness  ($(toc()))")

    return avg_happiness
end


function compute_happiness(sol, child_happiness, gift_happiness)
    # return compute_happiness_v2(sol, child_happiness, gift_happiness)
    return compute_happiness_v3(sol, child_happiness, gift_happiness)
end


"""
If sampling triplets or twins, sample together. Add constraint to ensure same
gift.
"""
function get_feasible_slice(sol=nothing; n_kids=12, n_gifts=2)
    version = "v3"

    if version == "v3"
        return get_feasible_slice_v3(sol, n_kids=n_kids, n_gifts=n_gifts)
    elseif verion =="v2"
        return get_feasible_slice_v2(sol, sample_size=n_kids)
    elseif verion =="v1"
        return get_feasible_slice_v1(sol, sample_size=n_kids)
    end
end



function get_feasible_slice_v1(sol=nothing;sample_size=10)
    sample_kids = rand([TRIPLETS_INDEX; TWINS_INDEX; SINGLE_INDEX], sample_size)

    triplet_ids = intersect(sample_kids, TRIPLETS_INDEX)
    twin_ids = intersect(sample_kids, TWINS_INDEX)
    for i in triplet_ids
        index = findfirst(sample_kids, i)
        splice!(sample_kids, index, i:i+2)
        # append!(triplets_index, index:index+2)
    end
    for i in twin_ids
        index = findfirst(sample_kids, i)
        splice!(sample_kids, index, i:i+1)
        # append!(twins_index, index:index+1])
    end

    return  convert(Array{Int64,1},sample_kids)
end



function get_feasible_slice_v2(sol=nothing;sample_size=10)
    rnd_draw = rand(1:100000)
    # @show rnd_draw
    if rnd_draw < 5000
        # sample_kids = rand([TRIPLETS_INDEX; TWINS_INDEX; SINGLE_INDEX], sample_size)
        sample_kids = rand(TRIPLETS_INDEX, sample_size)
    elseif rnd_draw < 45000
        # sample_kids = rand([TWINS_INDEX; SINGLE_INDEX], sample_size)
        sample_kids = rand([TRIPLETS_INDEX; TWINS_INDEX], sample_size)
    else
        # sample_kids = rand(SINGLE_INDEX, sample_size * 100)
        sample_kids = rand([TRIPLETS_INDEX; TWINS_INDEX; SINGLE_INDEX], sample_size)
    end


    triplet_ids = intersect(sample_kids, TRIPLETS_INDEX)
    twin_ids = intersect(sample_kids, TWINS_INDEX)
    for i in triplet_ids
        index = findfirst(sample_kids, i)
        splice!(sample_kids, index, i:i+2)
        # append!(triplets_index, index:index+2)
    end
    for i in twin_ids
        index = findfirst(sample_kids, i)
        splice!(sample_kids, index, i:i+1)
        # append!(twins_index, index:index+1])
    end

    return  convert(Array{Int64,1},sample_kids)
end


"""Get Feasible Samples v3
If sampling triplets or twins, sample together. Add constraint to ensure same
gift.

Improvements:
    - if sampling triplets, then sample atleast 3 single kids
        to enable swap.
    - sample based on gifts instead of kids

"""
function get_feasible_slice_v3(sol;n_kids=5, n_gifts=2)

    sample_gifts = choose_gifts(sample_size=n_gifts)

    sample_kids = []
    for i in sample_gifts
        current_sol = sol[findin(sol[:,2],[i]),:]
        curren_child_index = intersect(current_sol[:,1], [TRIPLETS_INDEX; TWINS_INDEX; SINGLE_INDEX])
        currrent_triplet_ids = intersect(curren_child_index, TRIPLETS_INDEX)
        current_twin_ids = intersect(curren_child_index, TWINS_INDEX)
        current_single_ids = intersect(curren_child_index, SINGLE_INDEX)

        for i in 1:n_kids
            rnd_draw = rand(1:100000)
            if rnd_draw < 5000
                draw = rand(currrent_triplet_ids, 1)
            elseif (rnd_draw < 45000) & (rnd_draw >= 5000)
                draw = rand([current_twin_ids], 1)
            else
                draw = rand(curren_child_index, 1)
            end
            sample_kids = vcat(sample_kids, rand(curren_child_index, n_kids))
        end
    end

    # rnd_draw = rand(1:100000)
    # if rnd_draw < 5000
    #     sample_kids = rand(currrent_triplet_ids, sample_size)
    # elseif rnd_draw < 45000
    #     sample_kids = rand([current_twin_ids, current_single_ids], sample_size)
    # else
    #     sample_kids = rand(curren_child_index, sample_size)
    # end

    triplet_ids = intersect(sample_kids, TRIPLETS_INDEX)
    twin_ids = intersect(sample_kids, TWINS_INDEX)

    for i in triplet_ids
        index = findfirst(sample_kids, i)
        splice!(sample_kids, index, i:i+2)
        # append!(triplets_index, index:index+2)
    end
    for i in twin_ids
        index = findfirst(sample_kids, i)
        splice!(sample_kids, index, i:i+1)
        # append!(twins_index, index:index+1])
    end

    return  convert(Array{Int64,1},sample_kids)
end

function get_feasible_slice_singles(sol;n_kids=5)

    sample_gifts = choose_gifts(sample_size=n_kids)

    sample_kids = []
    for i in sample_gifts
        current_sol = sol[findin(sol[:,2],[i]),:]
        curren_child_index = intersect(current_sol[:,1], [TRIPLETS_INDEX; TWINS_INDEX; SINGLE_INDEX])
        current_single_ids = intersect(curren_child_index, SINGLE_INDEX)

        if length(current_single_ids) > 0
            sample_kids = vcat(sample_kids, rand(current_single_ids, 1))
        end
    end

    return  convert(Array{Int64,1},sample_kids)
end

function choose_gifts(;sample_size=2)
    sample_gift_ids = rand(0:N_GIFT_TYPE,sample_size)
    return sample_gift_ids
end



function get_twin_index(sample_kids)
    twins_index = []
    twin_ids = intersect(sample_kids, TWINS_INDEX)
    for i in twin_ids
        index = findfirst(sample_kids, i)
        append!(twins_index, index)
    end

    return  twins_index
end


"""Return 1-based indices of 1-st triplets in sample
"""
function get_triplet_index(sample_kids)
    triplets_index = []
    triplet_ids = intersect(sample_kids, TRIPLETS_INDEX)
    for i in triplet_ids
        index = findfirst(sample_kids, i)
        append!(triplets_index, index)
    end

    return  triplets_index
end



function run_opt(init, child_happiness, gift_happiness,
                current_happiness, current_avg_happiness;
                history=[],  n_kids=100, n_gifts=2,
                single_multiplier=10,
                singles_only_mode=false,
                threshold_cutoff=1.)
    sol= copy(init)

    no_improvement = 0

    # info("round $i")

    # sample_kids = rand(1:N_CHILDREN,BLOCK_SIZE)
    # sample_kids, triplets, twins = get_feasible_slice()
    # sample_kids = get_feasible_slice(sample_size=sample_size)
    # sample_kids = get_feasible_slice_v2(sample_size=sample_size)

    # if singles_only_mode == true
    #     sample_kids = get_feasible_slice_singles(sol;n_kids=n_kids*single_multiplier)
    # else
    #     rnd_draw = rand(1:1e6)
    #     if rnd_draw < 50000
    #         sample_kids = get_feasible_slice(sol, n_kids=n_kids, n_gifts=n_gifts)
    #     else
    #         sample_kids = get_feasible_slice_singles(sol;n_kids=n_kids*single_multiplier)
    #     end
    # end

    # sample_kids = temp_feas_slice(sol)
    # sample_kids = get_feasible_slice_singles(sol;n_kids=n_kids*single_multiplier)

    sample_kids = sample_mixed(sol,current_happiness, n_kids=n_kids * single_multiplier, threshold_cutoff=threshold_cutoff)

    triplets = get_triplet_index(sample_kids)
    twins = get_twin_index(sample_kids)
    # @show twins
    # @show triplets

    solution_block = sol[sample_kids+1, :]
    current_kids = solution_block[:,1]
    current_gifts = solution_block[:,2]
    current_sample_happiness = current_happiness[sample_kids+1, :]

    # info("optimizing")
    tic()
    # kids, gifts = optimize_block(solution_block)
    try
        kids, gifts = optimize(
                            current_kids,
                            current_gifts,
                            child_happiness,
                            gift_happiness;
                            triplets=triplets,
                            twins=twins)
        # sum(compute_happiness_vectors(hcat(kids, gifts), child_happiness, gift_happiness))
        # sum(compute_happiness_vectors(hcat(current_kids, current_gifts), child_happiness, gift_happiness))

        # println("optimization time :", toc())
        # kids - current_kids

        new_sol = copy(sol)
        new_sol[kids+1,2] = gifts

        # hap_copy = copy(current_happiness)
        new_sample_happiness = compute_happiness_vectors(hcat(kids, gifts), child_happiness, gift_happiness)
        new_happiness = copy(current_happiness)
        new_happiness[kids+1, :] = new_sample_happiness
        new_avg_happiness = compute_avg_happiness(new_happiness)

        # if sum(new_sample_happiness) > sum(current_sample_happiness)
        if new_avg_happiness > current_avg_happiness
            if Utils.check_feas(new_sol) == true
                # info("sample happiness sum improved by: $(sum(new_sample_happiness) - sum(current_sample_happiness))")
                sol[kids+1,2] = gifts

                # current_avg_happiness = compute_avg_happiness(current_happiness)
                # new_avg_happiness = compute_avg_happiness(current_happiness)
                # info("happiness change: $current_avg_happiness -> $(new_avg_happiness)")
                current_avg_happiness = new_avg_happiness
                current_happiness[kids+1, :] = new_sample_happiness

            else
                info("new solution infeasible")
                no_improvement += 1
            end
        end
    catch e
        @show e
    end
    # compute_happiness(sol, child_happiness, gift_happiness)

    # info("rounds without improvement: $(no_improvement) of $n_rounds")
    return sol, current_happiness, current_avg_happiness
end




# function run_opt_parallel(init, child_happiness, gift_happiness; history=[], n_rounds=10, sample_size=1000)
#     # sol= copy(init)
#
#     sol = SharedArray{Int64}((N_CHILDREN,2), init=0)
#     sol[:,:] = init[:,:]
#
#
#     @parallel for i in 1:n_rounds
#         # info("round $i")
#
#         # sample_kids = rand(1:N_CHILDREN,BLOCK_SIZE)
#         # sample_kids, triplets, twins = get_feasible_slice()
#         # sample_kids = get_feasible_slice(sample_size=sample_size)
#         sample_kids = get_feasible_slice_v2(sample_size=sample_size)
#
#         triplets = get_triplet_index(sample_kids)
#         twins = get_twin_index(sample_kids)
#         # @show twins
#         # @show triplets
#
#         solution_block = sol[sample_kids+1, :]
#         current_kids = solution_block[:,1]
#         current_gifts = solution_block[:,2]
#
#         # info("optimizing")
#         tic()
#         # kids, gifts = optimize_block(solution_block)
#         try
#             kids, gifts = optimize(current_kids, current_gifts, child_happiness, gift_happiness; triplets=triplets, twins=twins)
#             # println("optimization time :", toc())
#
#             sol_copy = copy(sol)
#             sol_copy[kids+1,2] = gifts
#             if Utils.check_feas(sol_copy) == true
#                 sol[kids+1,2] = gifts
#             else
#                 info("new solution infeasible")
#             end
#         catch e
#             @show e
#         end
#
#         # compute_happiness(sol, child_happiness, gift_happiness)
#     end
#     return sol
# end



function main(;init="heuristic")
    # sample_size = 20
    sample_size_step = 5
    non_improving_rounds = 0


    n_kids = 50
    n_gifts = 3
    single_multiplier=10
    singles_only_mode = true
    threshold_cutoff = .5

    gift_pref, child_pref, gift_happiness, child_happiness = Preprocess.load_data()

    history = fill!(Array{Float64}(1),0)
    init = "csv"
    # init_file = "data/output/sub_0.73.csv"
    init_file = "data/output/sub_0.8923240047294622.csv"
    if init == "heuristic"
        sol = Heuristics.heuristic_greedy(gift_pref)
    elseif init == "csv"
        init_sol = CSV.read(joinpath(pwd(),init_file), nullable=false)
        sol = convert(Array, init_sol)
    end

    if Utils.check_feas(sol) != true
        throw("Heuristic solution infeasible")
    end

    # happiness = compute_happiness(sol, child_happiness, gift_happiness)

    current_happiness = compute_happiness_vectors(sol, child_happiness, gift_happiness)
    current_avg_happiness = compute_avg_happiness(current_happiness)
    append!(history, current_avg_happiness)

    # Combinatorics.nthperm(sol[:,1], 2)
    # perm = Combinatorics.Permutations(sol[:,1], 2)
    # for p in perm
    #     @show p
    # end

    # Utils.check_feas(sol)
    # n_kids = 20
    # n_gifts = 3
    # single_multiplier=10

    for i=1:1000
        info("round $i")
        sol, current_happiness, current_avg_happiness = run_opt(sol,
                                            child_happiness,
                                            gift_happiness,
                                            current_happiness,
                                            current_avg_happiness;
                                            history=history,
                                            n_kids=n_kids,
                                            n_gifts=n_gifts,
                                            single_multiplier=single_multiplier,
                                            singles_only_mode=singles_only_mode,
                                            threshold_cutoff=threshold_cutoff)
        # sol = run_opt_parallel(sol, child_happiness, gift_happiness; history=history, n_rounds=1000, sample_size=20)
        # happiness = compute_happiness(sol, child_happiness, gift_happiness)

        # recompute happiness vectors
        # current_happiness = compute_happiness_vectors(sol, child_happiness, gift_happiness)
        # happiness = compute_avg_happiness(current_happiness)
        append!(history, current_avg_happiness)
        happiness_gain = (history[end] - history[end-1])
        # println("happiness gain: $happiness_gain")
        info("happiness $(history[end-1]) -> $(history[end]) [$happiness_gain]")



        if happiness_gain <= 0.
            non_improving_rounds += 1
            info("happiness did not improve, consecutive non improving rounds = $non_improving_rounds")
            if (non_improving_rounds % 3 == 0) & (non_improving_rounds > 1)
                # non_improving_rounds = 0
                # n_gifts += 1
                warn("happiness did not improve for $non_improving_rounds consecutive rounds, updating params")
                threshold_cutoff += .05
                warn("increasing threshold_cutoff to $threshold_cutoff")

                if (non_improving_rounds % (3*3) == 0)
                    n_kids += sample_size_step
                    warn("increasing params: n_kids = $n_kids, n_gifts=$n_gifts")
                end

            end
            if happiness_gain < 0
                warn("happiness decreased")
            end
        else
            non_improving_rounds = 0
            output = convert(DataFrame, sol)
            names!(output, [:ChildId, :GiftId])
            filename = joinpath(pwd(),"data/output/sub_$(history[end]).csv")
            CSV.write(filename,output)
            info("solution saved at $filename")
        end
        if non_improving_rounds > 20
            warn("no improvements in $non_improving_rounds round, terminating")
            break
        end
    end
end


function temp_feas_slice(sol)
    n_gifts = 2
    n_kids = 10
    sample_gifts = choose_gifts(sample_size=n_gifts)

    current_sol = sol[findin(sol[:,2],sample_gifts),:]
    curren_child_index = intersect(current_sol[:,1], [TRIPLETS_INDEX; TWINS_INDEX; SINGLE_INDEX])
    currrent_triplet_ids = intersect(curren_child_index, TRIPLETS_INDEX)
    current_twin_ids = intersect(curren_child_index, TWINS_INDEX)
    current_single_ids = intersect(curren_child_index, SINGLE_INDEX)

    gifts_pref_kids = gift_pref[sample_gifts+1,:]
    intersect(gifts_pref_kids, curren_child_index)

    sample_kids = []
    sample_kids = vcat(sample_kids, rand(curren_child_index, n_kids))
    sample_kids = vcat(sample_kids, rand(gifts_pref_kids, n_kids))


    triplet_ids = intersect(sample_kids, TRIPLETS_INDEX)
    twin_ids = intersect(sample_kids, TWINS_INDEX)

    for i in triplet_ids
        index = findfirst(sample_kids, i)
        splice!(sample_kids, index, i:i+2)
        # append!(triplets_index, index:index+2)
    end
    for i in twin_ids
        index = findfirst(sample_kids, i)
        splice!(sample_kids, index, i:i+1)
        # append!(twins_index, index:index+1])
    end

    sample_kids =  convert(Array{Int64,1},sample_kids)

    return sample_kids
end

"""
Args:
    threshold_cutoff (float): a value which divides 50-50 sampling point
"""
function sample_mixed(sol,current_happiness; n_kids=20, threshold_cutoff=1.)
    # sample_gifts = choose_gifts(sample_size=n_kids)

    sorted_ix = sortperm(current_happiness[:,1])
    # current_happiness[sorted_ix,:]
    # sol[sorted_ix,:]

    # threshold_cutoff=.7
    sorted_single_kids = convert(Array{Int},intersect(sorted_ix-1, SINGLE_INDEX))
    # k = findfirst(x -> x >= 1., current_happiness[sorted_single_kids+1,1] )
    # k = findfirst(index(x) --> x >= 1., current_happiness[sorted_single_kids+1,1] )
    # threshold_cutoff=.7
    threshold = findfirst(current_happiness[sorted_single_kids+1,1], threshold_cutoff)
    # current_happiness[sorted_single_kids[k-5:k+5]+1,:]
    # current_happiness[sorted_single_kids+1,:]
    # sol[sorted_single_kids[1:threshold-1]+1,1]
    # current_happiness[sorted_single_kids[1:threshold-1]+1,1]

    # current_happiness[sorted_single_ix[1:convert(Int,round(length(sorted_single_ix) * threshold))]+1,:]

    # length(current_happiness[sorted_single_kids[1:threshold-1]+1,:])
    # current_happiness[sorted_single_kids[threshold:end]+1,:]
    # sample1 = StatsBase.sample(sorted_single_ix[1:convert(Int,round(length(sorted_single_kids) * threshold))],Int(n_kids/2))
    # sample2 = StatsBase.sample(sorted_single_ix[convert(Int,round(length(sorted_single_kids) * threshold))+1:end],Int(n_kids/2))

    sample1 = StatsBase.sample(sol[sorted_single_kids[1:threshold-1]+1,1],min(Int(n_kids/2),threshold), replace=false)
    sample2 = StatsBase.sample(sol[sorted_single_kids[threshold:end]+1,1],Int(n_kids/2), replace=false)


    # current_happiness[sample1+1,:]
    # sol[sample1+1,:]
    # current_happiness[sample_kids+1,:]
    # sol[sample1,:]


    # current_happiness[[sorted_ix<100000]]
    # current_happiness[sample_kids1+1,:]

    # sample_kids1 = get_feasible_slice_singles(
    #     sol[sorted_ix[1:convert(Int,N_CHILDREN * threshold)],:];
    #     n_kids=Int(n_kids/2))
    # sample_kids2 = get_feasible_slice_singles(
    #     sol[sorted_ix[convert(Int,N_CHILDREN * threshold)+1:N_CHILDREN],:];
    #     n_kids=Int(n_kids/2))
    sample_kids = [sample1;sample2]
    return sample_kids
end

# addprocs(6)
main()






# cost = zeros(1e6,1e3)
# for c in 1:N_CHILDREN
#     for g in 1:N_GIFT_TYPE
#         cost[c,g] = child_happiness[c-1][g-1] + gift_happiness[g-1][c-1]
#     end
# end



# Utils.avg_normalized_happiness(sol, child_pref, gift_pref)
# plot(history)

# write solution
# output = convert(DataFrame, sol)
# names!(output, [:ChildId, :GiftId])
# CSV.write(joinpath(pwd(),"data/output/sub_$(round(history[end],2)).csv"),output)





# /Users/mupadhye/git/mandarup/giftmatching/data/output/sub_0.73.csv

## load initial solution
# Utils.avg_normalized_happiness(sol, child_pref, gift_pref)
# init_df = CSV.read(joinpath(pwd(),"data/output/cpp_sub.csv"), header=true, nullable=false)
# init = convert(Array, init_df[2:end, :])
#
# sol = copy(init)
# Utils.check_feas(sol)
