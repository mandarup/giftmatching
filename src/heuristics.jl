
module Heuristics

using DataFrames
# using MLDataUtils
using CSV

include("Constants.jl")
using Constants

export make_dummy_output, heuristic_greedy, heuristic_dumb


# const N_CHILDREN = 1000000 # n children to give
# const N_GIFT_TYPE = 1000 # n types of gifts available
# const N_GIFT_QUANTITY = 1000 # each type of gifts are limited to this quantity
# const N_GIFT_PREF = 100 # number of gifts a child ranks
# const N_CHILD_PREF = 1000 # number of children a gift ranks
# const TWINS = ceil(0.04 * N_CHILDREN / 2.) * 2    # 4% of all population, rounded to the closest number
# const TRIPLETS = ceil(0.005 * N_CHILDREN / 3.) * 3    # 0.5% of all population, rounded to the closest number
# const RATIO_GIFT_HAPPINESS = 2
# const RATIO_CHILD_HAPPINESS = 2
#
# # index of first members of twins and triplets
# const TRIPLETS_INDEX = colon(0,3, TRIPLETS-1)
# const TWINS_INDEX = colon(TRIPLETS,2,TRIPLETS+TWINS-2)
# const TWINS_RANGE = [TRIPLETS, TRIPLETS+TWINS-1]
# const SINGLE_INDEX = colon(TRIPLETS + TWINS ,1,N_CHILDREN-1)
#
# # [i for i in TRIPLETS_INDEX]
# # [i for i in TWINS_INDEX]
#
#
#
#
# const WISHLIST_FILE = "data/input/child_wishlist_v2.csv"
# const GOODKIDS_FILE = "data/input/gift_goodkids_v2.csv"



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


function get_next_greedy_pref(gift_pref, gift_count, child_id; child_type=1)
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
# child_type = 2
# gift_count = fill!(Array{Int}(N_GIFT_QUANTITY),N_GIFT_QUANTITY)
# gift_id = find(x -> x >= child_type, gift_count)[1]
# get_next_greedy_pref(gift_count, 49000, child_type=1)
# gift_pref
#
# gift_pref[child_id + 1:child_id +  child_type, :]
# gift_pref[child_id + 1:child_id+child_type, :]


function heuristic_greedy(gift_pref)
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
        gift_id = get_next_greedy_pref(gift_pref,gift_count, child_id, child_type=3)
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
        gift_id = get_next_greedy_pref(gift_pref, gift_count, child_id, child_type=2)
        for i in 1:2
            output[child_id + i, 2] = gift_id
            gift_count[gift_id+1] -= 1
        end

    end

    for t in SINGLE_INDEX# TRIPLETS + TWINS + 1: N_CHILDREN -1
        child_id = convert(Int,t)
        gift_id = get_next_greedy_pref(gift_pref, gift_count, child_id, child_type=1)
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
    # println(gift_count_test)

    return output
end


# output = make_dummy_output()
# output = heuristic_dumb()
# # println(output)
# check_feas(output)
# score = avg_normalized_happiness(output, child_pref, gift_pref)
# println("normalized score dumb heuristic: $score")
#
# output = heuristic_greedy1()
# # println(output)
# check_feas(output)
# score = avg_normalized_happiness(output, child_pref, gift_pref)
# println("normalized score greedy heuristic: $score")


end
