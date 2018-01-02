

using DataStructures
using DataFrames
using MLDataUtils
using CSV
using JuMP
using JLD



include("src/AssignmentModel.jl")
include("src/Utils.jl")

# import AssignmentModel
# import SantaUtils



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

const PROCESSED_DATA_FILE = "data/working/data.jld"
const PROCESSED_DATA_PATH = joinpath(pwd(),PROCESSED_DATA_FILE)


addprocs(7)

gift_pref_df = CSV.read(joinpath(pwd(),WISHLIST_FILE), header=false, nullable=false)
child_pref_df= CSV.read(joinpath(pwd(),GOODKIDS_FILE), header=false, nullable=false)

#DataFrames.head(gift_pref_df)
#head(child_pref_df)

gift_pref = convert(Array, gift_pref_df[:, 2:end])
child_pref = convert(Array, child_pref_df[:, 2:end])
size(child_pref)
size(gift_pref)

gift_pref

clear!(:gift_pref_df)
clear!(:child_pref_df)


gift_happiness = Dict()
for g in range(0,N_GIFT_TYPE-1)
    gift_happiness[g] = DefaultDict( -1. )
    for (i, c) in enumerate(child_pref[g+1,:])
        #@show N_CHILD_PREF, - i, - 1, 2. * (N_CHILD_PREF - i - 1)
        gift_happiness[g][c] = 2. * (N_CHILD_PREF - i - 1)
    end
end


child_happiness = Dict()
for c in range(0,N_CHILDREN-1)
    child_happiness[c] = DefaultDict(-1.)
    for (i, g) in enumerate(gift_pref[c+1,:])
        child_happiness[c][g] = 2. * (N_GIFT_PREF - i - 1)
    end
end

# jldopen( PROCESSED_DATA_PATH, "w") do file
#     write(file, "child_pref", child_pref)  # alternatively, say "@write file A"
#     write(file, "gift_pref", gift_pref)
#     write(file, "gift_happiness", gift_happiness)
#     write(file, "child_happiness", child_happiness)
# end

jldopen( joinpath(pwd(), "data/working/processed.jld"), "w") do file
    write(file, "gift_happiness", gift_happiness)
end

jldopen( joinpath(pwd(), "data/working/processed.jld"), "w") do file
    write(file, "child_happiness", child_happiness)
end

# sgift_happiness = jldopen(joinpath(pwd(), "data/working/processed.jld", "r") do file
#     read(file, "gift_happiness")
# end
#
# child_happiness = jldopen(joinpath(pwd(), "data/working/processed.jld", "r") do file
#     read(file, "child_happiness")
# end

#
#
# GIFT_IDS = np.array([[g] * N_GIFT_QUANTITY for g in range(N_GIFT_TYPE)]).flatten()
#
#
function avg_normalized_happiness2(output, child_pref, gift_pref)
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
    for (c, g) in gift_assignment
        total_child_happiness +=  CHILD_HAPPINESS[c][g]
        total_gift_happiness[g] += GIFT_HAPPINESS[g][c]
    end
    nch = total_child_happiness / N_CHILDREN
    ngh = mean(total_gift_happiness) / N_GIFT_TYPE
    print('normalized child happiness', nch)
    print('normalized gift happiness', ngh)
    return nch + ngh
end
