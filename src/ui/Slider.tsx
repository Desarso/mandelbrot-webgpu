import type { Component } from "solid-js";
import styles from "../App.module.css";

export const Slider: Component<{
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  digits?: number;
  suffix?: string;
  onInput: (value: number) => void;
}> = (props) => (
  <div class={styles.field}>
    <label class={styles.fieldLabel}>
      <span>{props.label}</span>
      <span class={styles.value}>
        {props.value.toFixed(props.digits ?? 0)}
        {props.suffix ?? ""}
      </span>
    </label>
    <input
      class={styles.slider}
      type="range"
      min={props.min}
      max={props.max}
      step={props.step}
      value={props.value}
      onInput={(e) => props.onInput(+e.currentTarget.value)}
    />
  </div>
);

export const Toggle: Component<{
  label: string;
  value: boolean;
  onChange: (value: boolean) => void;
}> = (props) => (
  <div class={`${styles.field} ${styles.toggleRow}`}>
    <span>{props.label}</span>
    <button
      class={`${styles.switch} ${props.value ? styles.switchOn : ""}`}
      role="switch"
      aria-checked={props.value}
      aria-label={props.label}
      onClick={() => props.onChange(!props.value)}
    />
  </div>
);
