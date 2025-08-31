import docker, tarfile
from io import BytesIO

client = docker.APIClient()

# create container
container = client.create_container(
    "bash:5.2",
    stdin_open=True,
    command="bash",
)
client.start(container)

# attach stdin to container and send data
s = client.attach_socket(
    container, params={"stdin": 1, "stream": 1, "stdout": 1, "stderr": 1}
)

print(s)

while True:
    original_text_to_send = input("$") + "\n"
    if original_text_to_send == "exit\n":
        s.close()
        break
    else:
        s._sock.send(original_text_to_send.encode("utf-8"))
        msg = s._sock.recv(1024)
        print(msg)
        print(len(msg))
        print("==================")
        print(msg.decode()[8:])


print("We're done here")
client.stop(container)
client.wait(container)
client.remove_container(container)
