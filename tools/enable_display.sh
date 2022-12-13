cat << EOF

After Run this script, the screen should display rainbow lik the fakesink output.

EOF

modetest -M xlnx -D fd4a0000.zynqmp-display -s 43@41:1920x1080@AR24

