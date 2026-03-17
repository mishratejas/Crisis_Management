from master_agent.master_graph import master_graph


def run_system():

    print("\n===== CRISIS MANAGEMENT AI =====")

    while True:

        image = input("\nEnter satellite image path (or exit): ")

        if image == "exit":
            break

        try:

            master_graph.invoke({
                "satellite_image": image,
                "field_reports":   [],
                "dispatch_config": {
                    "send_sms":       True,
                    "generate_audio": True,
                    "language":       "English",
                }
            })

        except Exception as e:

            print("\n[ERROR]", e)


if __name__ == "__main__":
    run_system()