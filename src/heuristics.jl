


using DataFrames
using MLDataUtils
using CSV
using JuMP

include("model.jl")


const N_CHILDREN = 1000000 # n children to give
const N_GIFT_TYPE = 1000 # n types of gifts available
const N_GIFT_QUANTITY = 1000 # each type of gifts are limited to this quantity
const N_GIFT_PREF = 100 # number of gifts a child ranks
const N_CHILD_PREF = 1000 # number of children a gift ranks
const TWINS = ceil(0.04 * N_CHILDREN / 2.) * 2    # 4% of all population, rounded to the closest number
const TRIPLETS = ceil(0.005 * N_CHILDREN / 3.) * 3    # 0.5% of all population, rounded to the closest number
const RATIO_GIFT_HAPPINESS = 2
const RATIO_CHILD_HAPPINESS = 2

# index of first members of twins and triplets
const TRIPLETS_INDEX = colon(0,3, TRIPLETS-1)
const TWINS_INDEX = colon(TRIPLETS,2,TRIPLETS+TWINS-2)
const TWINS_RANGE = [TRIPLETS, TRIPLETS+TWINS-1]
const SINGLE_INDEX = colon(TRIPLETS + TWINS ,1,N_CHILDREN-1)

# [i for i in TRIPLETS_INDEX]
# [i for i in TWINS_INDEX]




const WISHLIST_FILE = "data/input/child_wishlist_v2.csv"
const GOODKIDS_FILE = "data/input/gift_goodkids_v2.csv"

gift_pref_df = CSV.read(joinpath(pwd(),WISHLIST_FILE), header=false, nullable=false)
child_pref_df = CSV.read(joinpath(pwd(),GOODKIDS_FILE), header=false, nullable=false)

head(gift_pref_df)
head(child_pref_df)



gift_pref = convert(Array, gift_pref_df[:, 2:end])
child_pref = convert(Array, child_pref_df[:, 2:end])
size(child_pref)
size(gift_pref)


clear!(:gift_pref_df)
clear!(:child_pref_df)

function make_dummy_output()
    output = Array{Int}((N_CHILDREN,2))

    for i in 1:size(output)[1]
        try
            output[i,1] = i - 1
            output[i,2] = (i-1) % N_GIFT_TYPE
        catch
            @show i
        end
    end
    return output
end

function check_feas(output)
    # check if triplets have the same gift
    for t1 in TRIPLETS_INDEX #colon(0,3, TRIPLETS-1)
        # @show convert(Int,t1)
        index = convert(Int,t1) +1
        triplet1 = output[index, 2]
        triplet2 = output[index+1,2]
        triplet3 = output[index+2,2]
        # println("$index:$triplet1, $(index+1):$triplet2, $(index+2):$triplet3")
        #throw(AssertionError("$(triplet1[1]) == $(triplet2[1]) and $(triplet2[1]) == $(triplet3[1])"))
        if (triplet1[1] != triplet2[1]) | (triplet2[1] != triplet3[1])
            return false
        end
    end

    # check if twins have the same gift
    for t1 in TWINS_INDEX #colon(TRIPLETS-1,2,TRIPLETS+TWINS-2)

        index = convert(Int,t1) +1
        twin1 = output[index, 2]
        twin2 = output[index+1,2]
        #println("$index:$twin1, $(index+1):$twin2")
        if (twin1[1] != twin2[1])
            return false
        end
    end
    return true
end

function avg_normalized_happiness(output, child_pref, gift_pref)
    # zip(output[:,1], output[:,2])
    gift_assignment = Dict(zip(output[:,1] , output[:,2]))
    gift_counts = Dict(v=>0 for (k,v) in gift_assignment )
    tic()
    for (kid, gift) in gift_assignment
        gift_counts[gift] += 1
        if gift_counts[gift] > N_GIFT_QUANTITY
            throw(AssertionError("quantity for $gift==$(gift_counts[gift]) >=$N_GIFT_QUANTITY"))
        end
    end
    println("couting gifts ",toc())
    println("checking feasibility: ", check_feas(output))

    max_child_happiness = N_GIFT_PREF * RATIO_CHILD_HAPPINESS
    max_gift_happiness = N_CHILD_PREF * RATIO_GIFT_HAPPINESS

    total_child_happiness = 0
    total_gift_happiness = zeros(N_GIFT_TYPE)

    tic()
    for (child_id, gift_id) in gift_assignment

        # check if child_id and gift_id exist
        # assert child_id < N_CHILDREN
        # assert gift_id < N_GIFT_TYPE
        # assert child_id >= 0
        # assert gift_id >= 0
        #println(size(gift_pref[child_id+1,:]))
        gift_rank = find(x -> x == gift_id, gift_pref[child_id+1,:])
        if length(gift_rank) == 0
           child_happiness = -1
        else
           child_happiness = (N_GIFT_PREF - gift_rank[1]) * RATIO_CHILD_HAPPINESS
        end

        child_rank = find(x -> x == child_id, child_pref[gift_id + 1,:])
        if length(child_rank) == 0
           gift_happiness = -1
        else
           gift_happiness = (N_CHILD_PREF - child_rank[1]) * RATIO_GIFT_HAPPINESS
        end

        total_child_happiness += child_happiness
        total_gift_happiness[gift_id+1] += gift_happiness

    end
    println("total_child_happiness ", total_child_happiness)
    println("total_gift_happiness ", sum(total_gift_happiness))

    println("normalized child happiness=",total_child_happiness/(N_CHILDREN *max_child_happiness) ,
        ", normalized gift happiness",mean(total_gift_happiness) / (max_gift_happiness*N_GIFT_QUANTITY))

    # to avoid float rounding error
    # find common denominator
    # NOTE: I used this code to experiment different parameters, so it was necessary to get the multiplier
    # Note: You should hard-code the multipler to speed up, now that the parameters are finalized
    denominator1 = N_CHILDREN*max_child_happiness
    denominator2 = N_GIFT_QUANTITY*max_gift_happiness*N_GIFT_TYPE
    common_denom = lcm(denominator1, denominator2)
    multiplier = common_denom / denominator1

    # # usually denom1 > demon2
    return ((total_child_happiness*multiplier) ^ 3 + (sum(total_gift_happiness) ^3)) / (common_denom ^ 3)


end


function heuristic_dumb()
    output = Array{Int}(N_CHILDREN,2)
    output[:,1] = 0:N_CHILDREN-1
    output[:,2] = -1

    child_id::Int = 0
    gift_id::Int = 0
    gift_count = fill!(Array{Int}(N_GIFT_TYPE),N_GIFT_QUANTITY)
    # println(gift_count)

    #gift_flag = fill!(Array{Int}(N_GIFT_QUANTITY),0)
    # check if triplets have the same gift
    for t in TRIPLETS_INDEX #colon(0,3, TRIPLETS-1)
        # @show convert(Int,t1)
        child_id = convert(Int,t)
        for i in 1:3
            output[child_id + i, 2] = gift_id
            gift_count[gift_id+1] -= 1
            # @show child_id, output[child_id + i, 2], gift_id, gift_count[gift_id+1]
        end
        if gift_count[gift_id+1] < 3
            gift_id += 1
        end
    end
    #println(output[1:child_id,:])
    # println("=> $child_id")
    #println(output[1:child_id,:])
    # println(gift_count)
    # println(output[1:5000,:])

    gift_id = 0
    while gift_count[gift_id+1] < 2
        gift_id += 1
    end

    for t in TWINS_INDEX
        # @show convert(Int,t1)
        child_id = convert(Int,t)
        for i in 1:2
            output[child_id + i, 2] = gift_id
            gift_count[gift_id+1] -= 1
            #@show child_id, output[child_id + i, 2], gift_id, gift_count[gift_id+1]
        end
        if gift_count[gift_id+1] < 2
            gift_id += 1
        end
    end
    # println(gift_count)
    # println(compute_gift_count(output))

    gift_id = 0
    # println(gift_count[gift_id+1] <= 0)
    while gift_count[gift_id+1] <= 0
        # @show gift_id, gift_count[gift_id+1]
        gift_id += 1
    end
    # @show gift_id, gift_count[gift_id+1]

    for t in SINGLE_INDEX# TRIPLETS + TWINS + 1: N_CHILDREN -1
        # @show convert(Int,t1)
        #@show t, child_id, gift_id

        child_id = convert(Int,t)
        output[child_id + 1, 2] = gift_id
        gift_count[gift_id+1] -= 1
        #@show gift_id
        #@show child_id, output[child_id + i, 2], gift_id, gift_count[gift_id+1]
        # catch e
        #     println("child_id", child_id)
        #     @show t
        #     @show child_id
        #     @show gift_id
        #     #@show t, child_id, output[child_id +1, 2], gift_id, gift_count[gift_id+1]
        #     throw(e)
        # end
        while (gift_id < N_GIFT_QUANTITY-1) & (gift_count[gift_id+1] <= 0)
            # @show gift_id, gift_count[gift_id+1]
            gift_id += 1
            # @show gift_id
        end
    end

    gift_count_test = fill!(Array{Int}(N_GIFT_TYPE),N_GIFT_QUANTITY)
    for i in 1:size(output)[1]
        if output[i,2] >= 0
            gift_count_test[output[i,2]+1] -=1
        else
            throw(output[i,:])
        end
    end
    # println(gift_count_test)

    return output
end


function compute_gift_count(output)
    gift_count_test = fill!(Array{Int}(N_GIFT_TYPE),N_GIFT_QUANTITY)
    for i in 1:size(output)[1]
        if output[i,2] >= 0
            gift_count_test[output[i,2]+1] -=1
        end
    end
    return gift_count_test
end


function get_next_greedy_pref(gift_count, child_id; child_type=1)
    preferred_gifts = gift_pref[child_id + 1:child_id + child_type, :]
    # println(preferred_gifts)
    # println(size(preferred_gifts))
    gift_id = nothing

    # first go over 1st prefs of each kid, then second .. so on
    for k in 1:size(preferred_gifts)[1]
        for g in 1:size(preferred_gifts)[2]
            next_gift_id = preferred_gifts[k,g]
            if gift_count[next_gift_id+1] >= child_type
                #println("choosing $k $g: $next_gift_id")
                gift_id = next_gift_id
                break
            end
        end
        if gift_id != nothing
            break
        end
    end
    # if no greedy gift found then assign first available gift
    if gift_id == nothing
        gift_id = find(x -> x >= child_type, gift_count)[1]-1
    end
    return gift_id
end
#
# child_id = 0
child_type = 2
gift_count = fill!(Array{Int}(N_GIFT_QUANTITY),N_GIFT_QUANTITY)
gift_id = find(x -> x >= child_type, gift_count)[1]
# get_next_greedy_pref(gift_count, 49000, child_type=1)
# gift_pref
#
# gift_pref[child_id + 1:child_id +  child_type, :]
# gift_pref[child_id + 1:child_id+child_type, :]


function heuristic_greedy1()
    output = Array{Int}(N_CHILDREN,2)
    output[:,1] = 0:N_CHILDREN-1
    output[:,2] = -1

    child_id::Int = 0
    gift_id::Int = 0
    gift_count = fill!(Array{Int}(N_GIFT_TYPE),N_GIFT_QUANTITY)
    # println(gift_count)

    #gift_flag = fill!(Array{Int}(N_GIFT_QUANTITY),0)
    # check if triplets have the same gift
    for t in TRIPLETS_INDEX #colon(0,3, TRIPLETS-1)
        # @show convert(Int,t1)
        child_id = convert(Int,t)
        gift_id = get_next_greedy_pref(gift_count, child_id, child_type=3)
        for i in 1:3
            output[child_id + i, 2] = gift_id
            gift_count[gift_id+1] -= 1
            # @show child_id, output[child_id + i, 2], gift_id, gift_count[gift_id+1]
        end
        # if gift_count[gift_id+1] < 3
        #     gift_id += 1
        # end
    end
    #println(output[1:child_id,:])
    # println("=> $child_id")
    #println(output[1:child_id,:])
    # println(gift_count)
    # println(output[1:5000,:])

    gift_id = 0
    while gift_count[gift_id+1] < 2
        gift_id += 1
    end

    for t in TWINS_INDEX
        # @show convert(Int,t1)
        child_id = convert(Int,t)
        gift_id = get_next_greedy_pref(gift_count, child_id, child_type=2)
        for i in 1:2
            output[child_id + i, 2] = gift_id
            gift_count[gift_id+1] -= 1
        end

    end

    for t in SINGLE_INDEX# TRIPLETS + TWINS + 1: N_CHILDREN -1
        child_id = convert(Int,t)
        gift_id = get_next_greedy_pref(gift_count, child_id, child_type=1)
        output[child_id + 1, 2] = gift_id
        gift_count[gift_id+1] -= 1

        # while (gift_id < N_GIFT_QUANTITY-1) & (gift_count[gift_id+1] <= 0)
        #     gift_id += 1
        # end
    end

    gift_count_test = fill!(Array{Int}(N_GIFT_TYPE),N_GIFT_QUANTITY)
    for i in 1:size(output)[1]
        if output[i,2] >= 0
            gift_count_test[output[i,2]+1] -=1
        else
            throw(output[i,:])
        end
    end
    println(gift_count_test)

    return output
end


# output = make_dummy_output()

output = heuristic_dumb()
# println(output)
check_feas(output)
score = avg_normalized_happiness(output, child_pref, gift_pref)
println("normalized score dumb heuristic: $score")

output = heuristic_greedy1()
# println(output)
check_feas(output)
score = avg_normalized_happiness(output, child_pref, gift_pref)
println("normalized score greedy heuristic: $score")
