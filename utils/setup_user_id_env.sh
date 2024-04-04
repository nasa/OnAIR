echo "USER_ID="`id -u` > .env
echo "GROUP_ID="`id -g` >> .env

# IP Address is needed to run the groundstation with Windows 10 for the Xserver host
echo "IP_ADDRESS="`ipconfig | grep "IPv4" | grep -o "[0-9]*\.[0-9]*\.[0-9]*\.[0-9]*" | head -n 1` >> .env
