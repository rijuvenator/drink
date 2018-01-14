#!/usr/bin/awk -f
# take out the end of line commas. Might not be necessary, but nice.
BEGIN {
	months["Jan"] = "01"
	months["Feb"] = "02"
	months["Mar"] = "03"
	months["Apr"] = "04"
	months["May"] = "05"
	months["Jun"] = "06"
	months["Jul"] = "07"
	months["Aug"] = "08"
	months["Sep"] = "09"
	months["Oct"] = "10"
	months["Nov"] = "11"
	months["Dec"] = "12"
}

{
	mo = months[$1]
	dt = $2
	printf "%s-%02d: ", mo, dt
	i = 4
	while ($(i)!="")
	{
		if ($(i) ~ /\[(B|L|W)\]/)
		{
			sub(/^\[/,"",$(i))
			sub(/\]$/,"",$(i))
			printf "%s ", $(i)
		}
		if ($(i) ~ /^\(.*\)/)
		{
			printf "%s ", $(i)
		}
		i += 1
	}
	print ""
}
