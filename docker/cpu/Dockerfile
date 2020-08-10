# Multistage docker build, requires docker 17.05

# builder stage
FROM gcc:10.1 as builder

COPY . /app

RUN cd /app && make all

# runtime stage
FROM debian:buster-slim

COPY --from=builder /app/VanitySearch /usr/bin/VanitySearch

ENTRYPOINT ["/usr/bin/VanitySearch"]
