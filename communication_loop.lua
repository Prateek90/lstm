
require 'io'
require 'nngraph'
require 'base.lua'
stringx = require('pl.stringx')
data = require 'data.lua'

data.validdataset(1)
data.testdataset(1)
data.traindataset(1)

core = torch.load('core.net')
g_disable_dropout(core)


function readline()
  local line = io.read("*line")
  if line == nil then error({code="EOF"}) end
  line = stringx.split(line)
  if tonumber(line[1]) == nil then error({code="init"}) end
  for i = 2,#line do
    if data.vocab_map[line[i]] == nil then error({code="vocab", word = line[i]}) end
  end
  return line
end




while true do
  print("Query: len word1 word2 etc")
  local ok, line = pcall(readline)
  if not ok then
    if line.code == "EOF" then
      break -- end loop
    elseif line.code == "vocab" then
      print("Word not in vocabulary, only 'foo' is in vocabulary: ", line.word)
    elseif line.code == "init" then
      print("Start with a number")
    else
      print(line)
      print("Failed, try again")
    end
  else
    --[[print("Thanks, I will print foo " .. line[1] .. " more times")
    for i = 1, line[1] do io.write('foo ') end
    io.write('\n')--]]
    
    -- Number of times the neural net will guess a next word
    num_loop = tonumber(line[1])
    -- Initialize start state
    current_state = {}
    for i = 1, 2 do current_state[i] = torch.zeros(20, 200) end
    
    current_word_pos = 2
    current_word = line[current_word_pos]
    initial = 0

    io.write(current_word.." ")

    while initial < num_loop do
	entry = data.vocab_map[current_word]
        x = torch.Tensor(20):fill(entry)
      	err, new_state, pred = unpack(core:forward({x, x, current_state})) --don't care about label, just put x again
        g_replace_table(current_state, new_state)

    	if current_word_pos + 1 > #line then
	      rand_idx = torch.multinomial(pred[1]:exp():float(), 1)[1]
	      current_word = data.inverse_vocab_map[rand_idx]
	      initial = initial + 1
	    else
	      current_word_pos = current_word_pos + 1
   	    current_word = line[current_word_pos]
	    end
	    io.write(current_word.." ")
      end
    print("")
  end
end
