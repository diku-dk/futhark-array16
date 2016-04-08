fun int max(int x, int y) =
  if x < y then y else x

fun int main([int,n] as) = reduceComm(max, 0, as)
