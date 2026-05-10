# Architecture

This section documents the design philosophy and internal structure of the codebase. Understanding the architecture will help you navigate the code, debug issues, and extend the framework with new models.

---

<div class="grid cards" markdown>

-   :material-sitemap:{ .lg .middle } __High-Level Overview__

    ---

    Module dependency graph, registry pattern, and design principles.

    [:octicons-arrow-right-24: Overview](overview.md)

-   :material-network:{ .lg .middle } __U-Net Architecture__

    ---

    Block types, skip connections, timestep injection, and configuration system.

    [:octicons-arrow-right-24: U-Net](unet.md)

-   :material-play-circle:{ .lg .middle } __Training Pipeline__

    ---

    End-to-end flow from CLI to trained model: config loading, training loop, checkpointing.

    [:octicons-arrow-right-24: Training Pipeline](training_pipeline.md)

</div>
