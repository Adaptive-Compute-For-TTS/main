"use client";

import { useEffect, useRef, useState, type FormEvent } from "react";
import Link from "next/link";
import { rememberRun } from "@/lib/run-history";
import { RunHistory } from "./RunHistory";
import { Arrow, BrandMark, Check } from "@/components/Icons";

const NFE_OPTIONS = [4, 20, 32] as const;
type Nfe = (typeof NFE_OPTIONS)[number];
type Submission = { status: "idle" | "pending" | "success" | "error"; message: string; runId?: string };

function responseError(data: unknown): string {
  if (data && typeof data === "object") {
    const result = data as { error?: unknown; detail?: unknown };
    if (typeof result.error === "string") return result.error;
    if (typeof result.detail === "string") return result.detail;
    if (Array.isArray(result.detail)) {
      const messages = result.detail.flatMap((item: unknown) => {
        if (item && typeof item === "object" && "msg" in item && typeof item.msg === "string") return [item.msg];
        return [];
      });
      if (messages.length) return messages.join(" ");
    }
  }
  return "The request could not be submitted. Please try again.";
}

export function Playground() {
  const [text, setText] = useState("");
  const [nfe, setNfe] = useState<Nfe>(20);
  const [submission, setSubmission] = useState<Submission>({ status: "idle", message: "" });
  const request = useRef<AbortController | null>(null);
  const pending = submission.status === "pending";

  useEffect(() => () => request.current?.abort(), []);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!text.trim() || request.current) return;

    const controller = new AbortController();
    request.current = controller;
    setSubmission({ status: "pending", message: "Submitting…" });
    try {
      const response = await fetch("/api/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: text.trim(), nfe }),
        signal: controller.signal,
      });
      const data: unknown = await response.json().catch(() => null);
      if (!response.ok) throw new Error(responseError(data));
      const runId = data && typeof data === "object" && "id" in data && typeof data.id === "string" ? data.id : null;
      if (!runId) throw new Error("The speech service did not return a run ID. Please check the service before retrying.");
      await rememberRun(data);
      setSubmission({ status: "success", message: `Run ${runId} queued.`, runId });
    } catch (failure) {
      if (controller.signal.aborted) return;
      setSubmission({
        status: "error",
        message: failure instanceof TypeError
          ? "Could not reach the speech service. Check your connection and try again."
          : failure instanceof Error ? failure.message : "The request could not be submitted. Please try again.",
      });
    } finally {
      request.current = null;
    }
  }

  return (
    <>
      <a href="#main" className="skip-link">Skip to text input</a>
      <header className="site-header">
        <div className="header-inner">
          <Link href="/" className="brand" aria-label="Adaptive Compute home">
            <BrandMark />
            <span>Adaptive Compute</span>
          </Link>
        </div>
      </header>

      <main id="main" className="shell main-shell">
        <div className="page-heading">
          <h1>Start with something to say.</h1>
          <p>Enter your text and choose an NFE budget.</p>
        </div>

        <form className="input-layout" onSubmit={submit} aria-busy={pending}>
          <div className="text-field">
            <label className="field-label" htmlFor="speech-text">Text</label>
            <textarea
              id="speech-text"
              name="text"
              placeholder="Type or paste your text here…"
              value={text}
              onChange={(event) => {
                setText(event.target.value);
                setSubmission({ status: "idle", message: "" });
              }}
              disabled={pending}
              spellCheck
            />
          </div>

          <fieldset className="nfe-field" disabled={pending}>
            <legend className="field-label">NFE budget</legend>
            <div className="nfe-options">
              {NFE_OPTIONS.map((option) => (
                <label className={`nfe-option${nfe === option ? " selected" : ""}`} key={option}>
                  <input
                    type="radio"
                    name="nfe"
                    value={option}
                    checked={nfe === option}
                    onChange={() => {
                      setNfe(option);
                      setSubmission({ status: "idle", message: "" });
                    }}
                  />
                  <span className="nfe-value">{option}</span>
                  <span className="selection-indicator" aria-hidden="true">
                    {nfe === option && <Check />}
                  </span>
                </label>
              ))}
            </div>
          </fieldset>
          <div className="text-actions">
            <div className="request-feedback">
              <p role="status">{submission.status === "pending" || submission.status === "success" ? submission.message : ""}</p>
              {submission.runId && <a className="run-receipt" href={`/api/runs?id=${encodeURIComponent(submission.runId)}`} target="_blank" rel="noreferrer">View saved run ↗</a>}
              {submission.status === "error" && <p role="alert" className="request-error">{submission.message}</p>}
            </div>
            <button
              type="submit"
              className="submit-button"
              disabled={!text.trim() || pending}
              aria-label={pending ? "Submitting request" : "Generate speech"}
              title="Generate speech"
            >
              <Arrow />
            </button>
          </div>
        </form>
      </main>
      <RunHistory />
    </>
  );
}
