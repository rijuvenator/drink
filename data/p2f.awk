#!/usr/bin/awk -f
# make sure the file has a blank line at the end
BEGIN {
	isM = 0
}

{
	if ($1 == "")
	{
		print ""
		isM = 0
		next
	}
	if (isM == 0)
	{
		isM = 1
#		printf "%-6s :", $0
		printf "%-6s -", $0
		next
	}
	else
# legacy
#	{
#		a=$1
#		$1=""
#		printf "%s%s", $0, " ("a"),"
#	}
	{
		a=$1
		$1=""
		t=$NF
		$NF=""
		sub(/\s$/,//,$0)
		printf "%s%s%s", $0, "["t"]", " ("a"),"
	}

}
