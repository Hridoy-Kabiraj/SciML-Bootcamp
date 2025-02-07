# Task 2
println("Hello julia!")

# Task 3
a = 5
b = 6
println(a + b)

# Task 4
arr1 = [1, 2, 3, 4, 5]
totals = 0 
for i in 1:5
    global totals += arr1[i]  
end

println(totals)  

println(sum(arr1))


# Task 5
function double(x)
    return 2*x
end
input = 5
print(double(input))