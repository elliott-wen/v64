#!/bin/sh
# X session: a coloured root window, the matchbox window manager (draggable
# window borders), and a terminal. Move the mouse, drag the window, type in
# xterm — exercises virtio-gpu redraws + virtio-input end to end.
xsetroot -solid steelblue
matchbox-window-manager &
exec xterm -geometry 90x28+60+60
