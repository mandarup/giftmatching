

using DataStructures
using DataFrames
using MLDataUtils
using CSV
using JLD




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



gift_pref_df = CSV.read(joinpath(pwd(),WISHLIST_FILE), header=false, nullable=false)

child_pref_df= CSV.read(joinpath(pwd(),GOODKIDS_FILE), header=false, nullable=false)

head(gift_pref_df)
head(child_pref_df)

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


jldopen( joinpath(pwd(), "data/working/data.jld"), "w") do file
    write(file, "child_pref", child_pref)  # alternatively, say "@write file A"
    write(file, "gift_pref", gift_pref)
end

jldopen( joinpath(pwd(), "data/working/processed.jld"), "w") do file
    write(file, "gift_happiness", gift_happiness)
    write(file, "child_happiness", child_happiness)
end
