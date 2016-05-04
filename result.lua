require ('nngraph')

local stringx = require('pl.stringx')
local file = require('pl.file')

local ptb_path = "./data/"

local testfn = ptb_path .. "ptb.test.txt"

local vocab_idx = 0
local vocab_map = {}


local inverse_vocab_map={}

model=torch.load('core.net')

current_state = {}
    for i = 1, 2 do current_state[i] = torch.zeros(20, 200) end
    
function g_f3(f)
    return string.format("%.3f", f)
end


local function load_data(fname)
    local data = file.read(fname)
    data = stringx.replace(data, '\n', '<eos>')
    data = stringx.split(data)
    --print(string.format("Loading %s, size of data = %d", fname, #data))
    local x = torch.zeros(#data)
    for i = 1, #data do
        if vocab_map[data[i]] == nil then
            vocab_idx = vocab_idx + 1
            vocab_map[data[i]] = vocab_idx
            inverse_vocab_map[vocab_idx] = data[i]
        end
        x[i] = vocab_map[data[i]]
    end
    return x
end


function g_enable_dropout(node)
    if type(node) == "table" and node.__typename == nil then
        for i = 1, #node do
            node[i]:apply(g_enable_dropout)
        end
        return
    end
    if string.match(node.__typename, "Dropout") then
        node.train = true
    end
end

function g_replace_table(to, from)
    assert(#to == #from)
    for i = 1, #to do
        to[i]:copy(from[i])
    end
end

function g_disable_dropout(node)
    if type(node) == "table" and node.__typename == nil then
        for i = 1, #node do
            node[i]:apply(g_disable_dropout)
        end
        return
    end
    if string.match(node.__typename, "Dropout") then
        node.train = false
    end
end

function reset_state(state)
    state.pos = 1
    --if model ~= nil and model.start_s ~= nil then
        --for d = 1, 2 * params.layers do
        for d = 1, 2 do
            model.start_s[d]:zero()
        end
    --end
end

function run_test()
    reset_state(state_test)
    --g_disable_dropout(model.rnns)
    local perp = 0
    local len = state_test.data:size(1)
    

    -- no batching here
    --g_replace_table(model.s[0], model.start_s)
    for i = 1, (len - 1) do
        local x = state_test.data[i]
        local y = state_test.data[i + 1]
        perp_tmp,new_state, pred = unpack(model:forward({x, y, current_state})) --don't care about label, just put x again
        g_replace_table(current_state, new_state)
        perp = perp + perp_tmp[1]
        --g_replace_table(model.s[0], model.s[1])
    end
    print("Test set perplexity : " .. g_f3(torch.exp(perp / (len - 1))))
    --g_enable_dropout(model.rnns)
end
	
local function testdataset(batch_size)
    if testfn then
        local x = load_data(testfn)
        x = x:resize(x:size(1), 1):expand(x:size(1), batch_size)
        return x
    end
end

	
state_test = {data=testdataset(20)}


g_disable_dropout(model)
local states = state_test
 
--[[for _, state in pairs(states) do
    reset_state(state)
end--]]


run_test()
