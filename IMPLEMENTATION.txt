# Source Sought Feature Implementation

## Objective

Add a checkbox labelled **Source Sought** to the scraper UI. When enabled, the
SAM.gov scraper searches Contract Opportunities with these Notice Type filters:

- Source Sought
- Solicitation
- Combined Synopsis/Solicitation

When disabled, the scraper retains its existing behavior without applying these
additional filters.

## Client

Add the checkbox beside the existing scraper controls. It must be unchecked by
default and use the exact visible label `Source Sought`.

Bind the checkbox to a boolean field named `source_sought` in the existing
scrape request or options object:

```text
source_sought: boolean = false
```

Send the field through the same action or request that starts the scraper. Do
not trigger a scrape merely by toggling the checkbox.

The client must use the checkbox's current boolean value when it constructs the
scrape request. It must not serialize the value as `"true"`, `"false"`, the
visible label, or an omitted truthy value. If the client maintains form state,
initialize `source_sought` to `false` in that state and reset it consistently
with the other scraper controls.

### Client tests

Use the client project's existing component and request-mocking test tools. The
tests must locate the checkbox by its accessible role and exact label rather
than by a CSS class or DOM position.

1. Render the scraper form and assert that
   `getByRole("checkbox", { name: "Source Sought", exact: true })` is visible
   and unchecked.
2. Start a scrape without interacting with the checkbox and assert that the
   existing scrape action is called once with `source_sought: false`.
3. Check the control, assert that it is checked, start the scrape, and assert
   that the action is called once with `source_sought: true`.
4. Toggle the checkbox without starting a scrape and assert that no scrape
   action or network request occurs.
5. Uncheck it again, start the scrape, and assert that the submitted value is
   `false`.
6. If the form has a reset action, check the control, reset the form, and assert
   that the checkbox returns to unchecked.

These assertions must be added to the client test suite and pass alongside the
existing scraper-form regression tests.

## Backend implementation

Accept `source_sought` as an optional boolean in the existing scraper entry
point. Treat a missing value as `false` to preserve compatibility with existing
callers.

Keep the current default workflow unchanged when `source_sought` is false. When
it is true, complete the following setup before collecting results:

1. Open SAM.gov.
2. Navigate to **Search**.
3. Select **Contracting**.
4. Open **Contract Opportunities**.
5. Open **Filters**.
6. Expand **Notice Type** if it is collapsed.
7. Select **Source Sought**.
8. Select **Solicitation**.
9. Select **Combined Synopsis/Solicitation**.
10. Apply the filters or execute the search.
11. Wait for the filtered results to finish loading.
12. Collect all matching opportunities through the existing pagination or
    scrolling logic.
13. Pass the collected opportunities through the existing parsing,
    normalization, deduplication, storage, and return workflow.

Implement the filter setup as one reusable backend operation. Use stable
accessible labels, roles, test IDs, or SAM.gov filter values instead of element
positions or brittle generated CSS classes. Make selection idempotent: if a
required notice type is already selected, do not toggle it off.

## State flow

The feature should follow a single boolean from the UI to browser automation:

```text
Source Sought checkbox
        │
        ▼
source_sought request option
        │
        ▼
existing scraper entry point
        │
        ├── false ──> existing/default SAM.gov workflow
        │
        └── true  ──> apply three Notice Type filters
                            │
                            ▼
                    existing result workflow
```

Do not create a separate result format, database table, or persistence path for
Source Sought mode. The option changes only the SAM.gov search setup.

## Error handling

If Source Sought mode is enabled and a required filter cannot be found or
selected, stop the scrape with a descriptive error identifying the missing
Notice Type. Do not silently continue with an unfiltered or partially filtered
search.

Use the scraper's existing timeout, retry, logging, and browser cleanup
mechanisms. Do not catch an error unless it can be enriched and re-raised or
handled by the established workflow.

## Verification

Add automated coverage for the following behavior. The Client tests above are
required and must run in the existing client test command rather than in a
standalone or manually executed script:

1. The UI renders a `Source Sought` checkbox.
2. The checkbox is unchecked by default.
3. Starting a scrape while unchecked sends `source_sought: false` and uses the
   existing workflow.
4. Starting a scrape while checked sends `source_sought: true`.
5. A missing backend value defaults to false.
6. False mode does not invoke the Notice Type filter operation.
7. True mode selects exactly these values:
   - Source Sought
   - Solicitation
   - Combined Synopsis/Solicitation
8. An already-selected notice type remains selected.
9. A missing required filter produces a descriptive failure rather than an
   unfiltered scrape.
10. Filtered opportunities use the existing result parsing and persistence
    workflow.
11. The existing unchecked/default scraper tests continue to pass.

Use a fixture or mocked SAM.gov page for deterministic browser tests. Verify
live selectors separately before release because SAM.gov markup may change.

## Acceptance criteria

- The checkbox label is exactly **Source Sought**.
- The checkbox is opt-in and defaults to unchecked.
- Unchecked mode behaves exactly as it did before this feature.
- Checked mode applies all three required Notice Type filters before scraping.
- No filtered search runs when only a subset of the required filters was
  successfully selected.
- Results are returned and stored through the existing scraper workflow.
- UI, backend, browser automation, and regression tests pass.
